# app.py
import streamlit as st
import simpy, random, numpy as np, matplotlib.pyplot as plt, matplotlib.animation as animation
from io import BytesIO
import pandas as pd
import math

st.set_page_config(page_title="Retail Checkout Simulator (Upgraded)", layout="wide")

# -------------------------
# Utility distributions and helpers
# -------------------------
def sample_basket_size(dist_name, mean, sigma):
    if dist_name == "poisson":
        return max(1, np.random.poisson(mean))
    elif dist_name == "normal":
        return max(1, int(round(np.random.normal(mean, sigma))))
    else:
        return max(1, int(round(mean)))

def sample_service_time_from_basket(basket_size, scan_time=0.5, base_overhead=0.5):
    # simple model: overhead + per-item scan time (in minutes)
    return base_overhead + basket_size * scan_time

def time_varying_rate(base_rate, pattern, t_minute):
    # t_minute in minutes from start of sim
    # patterns: flat, morning_peak, double_peak
    if pattern == "flat":
        return base_rate
    if pattern == "morning_peak":
        # gaussian peak at 2 hours
        peak = base_rate * 2.5
        mu = 120
        sigma = 60
        factor = 1 + (math.exp(-0.5 * ((t_minute - mu) / sigma) ** 2) * 1.5)
        return base_rate * factor
    if pattern == "double_peak":
        # morning and evening-ish peaks
        mu1, mu2 = 90, 300
        sigma = 50
        factor = 1 + 1.2*math.exp(-0.5 * ((t_minute - mu1) / sigma) ** 2) + 1.2*math.exp(-0.5 * ((t_minute - mu2) / sigma) ** 2)
        return base_rate * factor
    return base_rate

# -------------------------
# Simulation core
# -------------------------
class CheckoutSim:
    def __init__(self, sim_minutes=240, base_arrival_rate=200, arrival_pattern="flat",
                 num_checkouts=4, express_count=0, express_max_items=8,
                 cashier_speeds=None, basket_mean=10, basket_sigma=3,
                 patience=None, seed=None, service_scan_time=0.3):
        self.sim_minutes = sim_minutes
        self.base_arrival_rate = base_arrival_rate
        self.arrival_pattern = arrival_pattern
        self.num_checkouts = num_checkouts
        self.express_count = express_count
        self.express_max_items = express_max_items
        self.cashier_speeds = cashier_speeds or [1.0]*num_checkouts
        self.basket_mean = basket_mean
        self.basket_sigma = basket_sigma
        self.patience = patience  # in minutes, or None
        self.service_scan_time = service_scan_time
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # resources: index 0..num_checkouts-1
        # model as separate Resources (capacity=1) to track each server
        self.env = simpy.Environment()
        self.checkouts = [simpy.Resource(self.env, capacity=1) for _ in range(num_checkouts)]

        # For the single shared-queue policy we also create a shared Resource that has
        # capacity = num_checkouts. This is a simple, reliable way to implement a single queue
        # feeding multiple servers (no fragile AnyOf/polling required).
        self.shared_queue = simpy.Resource(self.env, capacity=num_checkouts)

        # optional: express checkouts are the first express_count indices
        self.events = []  # per-customer event log (dicts)

    def run(self):
        self.env.process(self._arrivals_proc())
        self.env.run(until=self.sim_minutes)
        # Return a DataFrame of recorded events (may have NaNs for some columns)
        if len(self.events) == 0:
            return pd.DataFrame()
        return pd.DataFrame(self.events)

    def _arrivals_proc(self):
        cid = 0
        while True:
            t = self.env.now
            # arrival rate per hour -> convert to per minute; but allow time-varying
            current_rate = time_varying_rate(self.base_arrival_rate, self.arrival_pattern, t)
            lam_min = max(1e-6, current_rate / 60.0)
            inter = random.expovariate(lam_min)
            yield self.env.timeout(inter)
            cid += 1
            # spawn customer
            self.env.process(self._customer_proc(cid))

            if self.env.now >= self.sim_minutes:
                break

    def _choose_checkout_index(self, basket_size, policy="single"):
        # return checkout index chosen
        # policy: single (all share), multiple_random, multiple_shortest
        if policy == "single":
            return None
        if policy == "multiple_random":
            return random.randrange(self.num_checkouts)
        if policy == "multiple_shortest":
            lens = [len(r.queue) + r.count for r in self.checkouts]
            return int(np.argmin(lens))
        # fallback random
        return random.randrange(self.num_checkouts)

    def _customer_proc(self, cid):
        arrival = self.env.now
        basket = sample_basket_size("normal", self.basket_mean, self.basket_sigma)
        # choose if express-eligible
        is_express_eligible = (self.express_count > 0 and basket <= self.express_max_items)
        chosen_policy = st.session_state.get("queue_policy", "single") if "queue_policy" in st.session_state else "single"

        # routing for express: prefer express servers if eligible
        if is_express_eligible and self.express_count > 0:
            # choose among express servers using shortest
            express_indices = list(range(min(self.express_count, self.num_checkouts)))
            lens = [len(self.checkouts[i].queue) + self.checkouts[i].count for i in express_indices]
            idx = express_indices[int(np.argmin(lens))]
        else:
            idx = self._choose_checkout_index(basket, policy=chosen_policy)

        # If idx is None -> single shared queue behavior: use the shared_queue Resource.
        if idx is None:
            # request the shared queue (capacity = num_checkouts)
            req = self.shared_queue.request()
            if self.patience:
                res = yield req | self.env.timeout(self.patience)
                if req not in res:
                    # customer abandoned while waiting for the shared queue
                    self.events.append({
                        "customer_id": cid,
                        "arrival": arrival,
                        "basket": basket,
                        "abandoned": True,
                        "abandon_time": self.env.now
                    })
                    return
            else:
                yield req

            # Figure out which physical server to use once we've acquired the shared queue slot.
            # We'll pick the server with the smallest queue+busy count at this instant.
            server_idx = int(np.argmin([len(r.queue) + r.count for r in self.checkouts]))

            # Now request that specific physical server (this should be immediate in most cases
            # because shared_queue capacity ensured a server is available soon; but still request it).
            serv_req = self.checkouts[server_idx].request()
            yield serv_req

            # record start
            start_service = self.env.now
            base_service = sample_service_time_from_basket(basket, scan_time=self.service_scan_time)
            speed = self.cashier_speeds[server_idx] if server_idx < len(self.cashier_speeds) else 1.0
            service_time = base_service / max(1e-6, speed)

            self.events.append({
                "customer_id": cid,
                "arrival": arrival,
                "basket": basket,
                "assigned_server": server_idx,
                "start_service": start_service,
                "abandoned": False
            })

            yield self.env.timeout(service_time)
            end_service = self.env.now

            self.events.append({
                "customer_id": cid,
                "end_service": end_service,
                "assigned_server": server_idx
            })

            # release both the physical server and the shared queue slot
            self.checkouts[server_idx].release(serv_req)
            self.shared_queue.release(req)

        else:
            # regular per-server request flow (multiple queues scenario)
            req = self.checkouts[idx].request()
            # impatience:
            if self.patience:
                res = yield req | self.env.timeout(self.patience)
                if req not in res:
                    # abandoned
                    self.events.append({
                        "customer_id": cid,
                        "arrival": arrival,
                        "basket": basket,
                        "abandoned": True,
                        "abandon_time": self.env.now
                    })
                    return
            else:
                yield req
            server_idx = idx

            # got service on server_idx
            start_service = self.env.now
            base_service = sample_service_time_from_basket(basket, scan_time=self.service_scan_time)
            speed = self.cashier_speeds[server_idx] if server_idx < len(self.cashier_speeds) else 1.0
            service_time = base_service / max(1e-6, speed)

            self.events.append({
                "customer_id": cid,
                "arrival": arrival,
                "basket": basket,
                "assigned_server": server_idx,
                "start_service": start_service,
                "abandoned": False
            })

            yield self.env.timeout(service_time)
            end_service = self.env.now

            self.events.append({
                "customer_id": cid,
                "end_service": end_service,
                "assigned_server": server_idx
            })

            # release physical server
            self.checkouts[server_idx].release(req)

# -------------------------
# Animation helper (build frames from events)
# -------------------------
def build_customer_timelines(df_events):
    # convert events list into per-customer records with arrival, start, end, basket, server
    records = {}
    for _, row in df_events.iterrows():
        # row is a Series; use .get-like access via .to_dict
        d = row.to_dict()
        cid = d.get("customer_id")
        if cid is None:
            continue
        rec = records.setdefault(int(cid), {"customer_id": int(cid)})
        if "arrival" in d and pd.notna(d.get("arrival")):
            rec["arrival"] = float(d.get("arrival"))
        if "basket" in d and pd.notna(d.get("basket")):
            rec["basket"] = int(d.get("basket"))
        if "abandoned" in d and d.get("abandoned", False):
            rec["abandoned"] = True
            rec["abandon_time"] = float(d.get("abandon_time", rec.get("arrival", None)))
        if "start_service" in d and pd.notna(d.get("start_service")):
            rec["start"] = float(d.get("start_service"))
            rec["assigned_server"] = int(d.get("assigned_server", -1))
        if "end_service" in d and pd.notna(d.get("end_service")):
            rec["end"] = float(d.get("end_service"))
    # produce list sorted by customer id
    return [records[k] for k in sorted(records.keys())]

def make_animation(timelines, sim_minutes, num_checkouts, fps=6, interval_sec=0.5):
    # frames every interval (minutes)
    frame_step = 0.5  # minutes per frame
    times = np.arange(0, sim_minutes + frame_step, frame_step)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 30)
    ax.set_ylim(-1, num_checkouts + 1)
    ax.axis('off')
    server_x = 22
    server_y = np.arange(num_checkouts)

    # draw server boxes
    for i, y in enumerate(server_y):
        ax.add_patch(plt.Rectangle((server_x-1.5, y-0.35), 3.0, 0.7, fill=False))
        ax.text(server_x+1.8, y, f"S{i}", va="center")

    def get_positions_at(t):
        xs, ys, cs = [], [], []
        # compute waiting list at time t
        waiting = []
        for rec in timelines:
            arrival = rec.get("arrival", 1e9)
            start = rec.get("start", None)
            end = rec.get("end", None)
            abandoned = rec.get("abandoned", False)
            abandon_time = rec.get("abandon_time", 1e9)
            if arrival <= t and not abandoned and (start is None or start > t):
                waiting.append(rec)
        for rec in timelines:
            arrival = rec.get("arrival", None)
            if arrival is None or arrival > t:
                continue
            if rec.get("abandoned", False) and rec.get("abandon_time", 1e9) <= t:
                xs.append(0)
                ys.append(-0.7)
                cs.append('grey')
                continue
            start = rec.get("start", None)
            end = rec.get("end", None)
            server = rec.get("assigned_server", None)
            if start is not None and start <= t and (end is None or t < end):
                xs.append(server_x)
                ys.append(server)
                cs.append('tab:blue')
            elif end is not None and t >= end:
                xs.append(server_x + 4 + 0.2 * (t - end))
                ys.append(server)
                cs.append('tab:green')
            else:
                # waiting in queue: position based on waiting index
                try:
                    pos = waiting.index(rec)
                except ValueError:
                    pos = 0
                x = 2 + pos * 1.0
                y = (pos % max(1, num_checkouts))
                xs.append(x)
                ys.append(y)
                cs.append('tab:orange')
        return xs, ys, cs

    def update(frame_i):
        t = times[frame_i]
        xs, ys, cs = get_positions_at(t)
        ax.collections.clear()
        ax.scatter(xs, ys, s=80, c=cs)
        ax.set_title(f"Sim minute {t:.1f}")
        return ax.collections

    anim = animation.FuncAnimation(fig, update, frames=len(times), blit=False, repeat=False)
    buf = BytesIO()
    anim.save(buf, writer='pillow', fps=6)
    plt.close(fig)
    buf.seek(0)
    return buf

# -------------------------
# Streamlit UI & control
# -------------------------
st.sidebar.title("Simulation controls")

sim_hours = st.sidebar.slider("Sim length (hours)", 0.5, 12.0, 4.0, 0.5)
sim_minutes = int(sim_hours * 60)
base_rate = st.sidebar.slider("Base arrival rate (customers/hour)", 10, 1000, 200, 10)
arrival_pattern = st.sidebar.selectbox("Arrival time pattern", ["flat", "morning_peak", "double_peak"])
queue_policy = st.sidebar.selectbox("Queue policy", ["single", "multiple_shortest", "multiple_random"])
# store chosen policy in session state so simulation code can read it
st.session_state["queue_policy"] = queue_policy

num_checkouts = st.sidebar.slider("Number of checkouts", 1, 12, 4)
express_count = st.sidebar.slider("Number of express checkouts (first N)", 0, num_checkouts, 1)
express_max_items = st.sidebar.slider("Express max items", 1, 20, 8)
st.sidebar.write("---")
st.sidebar.write("Cashier speeds (multiplier)")
cashier_speeds = []
for i in range(num_checkouts):
    default = 1.0
    cashier_speeds.append(st.sidebar.number_input(f"Speed {i+1}", 0.2, 3.0, default, 0.1, key=f"spd{i}"))

st.sidebar.write("---")
st.sidebar.write("Customer properties")
basket_mean = st.sidebar.number_input("Avg basket size (items)", 1, 100, 12)
basket_sigma = st.sidebar.number_input("Basket sigma (items)", 0.0, 50.0, 3.0)
patience = st.sidebar.number_input("Patience (minutes, 0 = no patience)", 0.0, 999.0, 0.0)
patience_val = None if patience <= 0 else float(patience)

st.sidebar.write("---")
st.sidebar.write("Other")
scan_time = st.sidebar.number_input("Per-item scan time (minutes)", 0.05, 1.0, 0.25, 0.01)
seed = st.sidebar.number_input("Random seed", 0, 9999999, 42)
run_button = st.sidebar.button("Run simulation")

st.title("Retail Checkout Simulator — Upgraded")
st.markdown("Features: express lanes, per-cashier speeds, patience, time-of-day arrivals, CSV export, and an animation.")

if run_button:
    with st.spinner("Running simulation..."):
        sim = CheckoutSim(sim_minutes=sim_minutes, base_arrival_rate=base_rate,
                          arrival_pattern=arrival_pattern, num_checkouts=num_checkouts,
                          express_count=express_count, express_max_items=express_max_items,
                          cashier_speeds=cashier_speeds, basket_mean=basket_mean,
                          basket_sigma=basket_sigma, patience=patience_val,
                          seed=int(seed), service_scan_time=scan_time)
        df = sim.run()

    st.subheader("Simulation summary")
    # Process dataframe into per-customer aggregated rows
    if df.empty:
        st.write("No events recorded.")
    else:
        timelines = build_customer_timelines(df)
        rows = []
        for rec in timelines:
            rows.append({
                "customer_id": rec.get("customer_id"),
                "arrival": rec.get("arrival"),
                "start": rec.get("start"),
                "end": rec.get("end"),
                "assigned_server": rec.get("assigned_server"),
                "basket": rec.get("basket"),
                "abandoned": rec.get("abandoned", False)
            })
        summary_df = pd.DataFrame(rows)
        # compute metrics
        served = summary_df[~summary_df["abandoned"] & summary_df["end"].notnull()]
        avg_wait = served["start"].sub(served["arrival"]).mean() if not served.empty else 0.0
        median_wait = served["start"].sub(served["arrival"]).median() if not served.empty else 0.0
        p95_wait = served["start"].sub(served["arrival"]).quantile(0.95) if not served.empty else 0.0
        throughput = len(served) * 60.0 / sim_minutes

        st.metric("Customers served", len(served))
        st.metric("Abandoned customers", int(summary_df["abandoned"].sum()) if "abandoned" in summary_df else 0)
        st.metric("Avg wait (min)", f"{avg_wait:.2f}")
        st.metric("Median wait (min)", f"{median_wait:.2f}")
        st.metric("Throughput (cust/hr)", f"{throughput:.1f}")

        st.download_button("Download events CSV", df.to_csv(index=False).encode('utf-8'),
                           file_name="events.csv", mime="text/csv")

        st.dataframe(summary_df.sort_values("arrival").head(200))

        # Animation (build and display)
        with st.spinner("Building animation..."):
            timelines_list = timelines
            try:
                gif_buf = make_animation(timelines_list, sim_minutes=sim_minutes, num_checkouts=num_checkouts)
                st.image(gif_buf.getvalue(), caption="Simulation animation (schematic)")
            except Exception as e:
                st.warning(f"Animation failed: {e}")

        # Small charts
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        # wait histogram
        waits = served["start"].sub(served["arrival"]).dropna() if not served.empty else []
        ax[0].hist(waits, bins=30)
        ax[0].set_title("Wait time distribution (served)")
        ax[0].set_xlabel("Minutes")

        # service time histogram
        services = served["end"].sub(served["start"]).dropna() if not served.empty else []
        ax[1].hist(services, bins=30)
        ax[1].set_title("Service time distribution (served)")
        ax[1].set_xlabel("Minutes")

        st.pyplot(fig)
