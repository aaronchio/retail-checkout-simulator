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
        # optional: express checkouts are the first express_count indices
        self.events = []  # per-customer event log (dicts)

    def run(self):
        self.env.process(self._arrivals_proc())
        self.env.run(until=self.sim_minutes)
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
            # special marker for single: None and we request a combined semaphore
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
        chosen_policy = st.session_state.get("queue_policy", "single")

        # routing for express: prefer express servers if eligible
        if is_express_eligible and self.express_count > 0:
            # choose among express servers using shortest
            express_indices = list(range(min(self.express_count, self.num_checkouts)))
            lens = [len(self.checkouts[i].queue) + self.checkouts[i].count for i in express_indices]
            idx = express_indices[int(np.argmin(lens))]
        else:
            idx = self._choose_checkout_index(basket, policy=chosen_policy)

        # If idx is None -> use single shared resource (simulate by picking the server that becomes free earliest)
        # To implement single-queue behavior with separate resources, we place in the *global queue* conceptually:
        if idx is None:
            # implement manual queue: wait until any server free by registering a dummy request
            # We'll wait until any of the servers becomes available: use events
            reqs = [r.request() for r in self.checkouts]
            # optionally implement patience
            if self.patience:
                results = yield simpy.events.AnyOf(self.env, reqs) | self.env.timeout(self.patience)
                if isinstance(results, simpy.events.Event) and results.triggered and results.value == None:
                    # timeout case
                    # mark abandon
                    self.events.append({
                        "customer_id": cid,
                        "arrival": arrival,
                        "basket": basket,
                        "abandoned": True,
                        "abandon_time": self.env.now
                    })
                    # release any granted requests (shouldn't be any)
                    for r in reqs:
                        try:
                            r.cancel()
                        except Exception:
                            pass
                    return
                # find which request was granted
                chosen = None
                for r in reqs:
                    if r in results:
                        chosen = r
                        break
                server_idx = None
                # determine which server corresponds to the granted request by checking counts (granted reduces queue)
                for i, rsrc in enumerate(self.checkouts):
                    if rsrc.count > 0:
                        server_idx = i
                        break
                if server_idx is None:
                    server_idx = 0
            else:
                # wait for first available checkout
                # block until any server free
                while True:
                    # try to request every server non-blocking
                    for i, r in enumerate(self.checkouts):
                        req = r.request()
                        got = req in (yield simpy.events.AnyOf(self.env, [req]) )
                        # this approach is awkward in SimPy; simplify: wait until any server free by polling
                    # simpler approach: pick the server with earliest expected availability by using queue length
                    lens = [len(r.queue) + r.count for r in self.checkouts]
                    server_idx = int(np.argmin(lens))
                    req = self.checkouts[server_idx].request()
                    yield req
                    break
        else:
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
        # compute service time from basket and cashier speed
        base_service = sample_service_time_from_basket(basket, scan_time=self.service_scan_time)
        speed = self.cashier_speeds[server_idx] if server_idx < len(self.cashier_speeds) else 1.0
        service_time = base_service / max(1e-6, speed)
        # record start
        self.events.append({
            "customer_id": cid,
            "arrival": arrival,
            "basket": basket,
            "assigned_server": server_idx,
            "start_service": start_service,
            "abandoned": False
        })
        # simulate
        yield self.env.timeout(service_time)
        end_service = self.env.now
        # record completion (append another event row for completion)
        self.events.append({
            "customer_id": cid,
            "end_service": end_service,
            "assigned_server": server_idx
        })
        # release resource
        self.checkouts[server_idx].release(req)


# -------------------------
# Animation helper (build frames from events)
# -------------------------
def build_customer_timelines(df_events):
    # convert events list into per-customer records with arrival, start, end, basket, server
    records = {}
    for _, row in df_events.iterrows():
        # events created with different keys; make robust
        cid = row.get("customer_id")
        if cid is None:
            continue
        rec = records.setdefault(cid, {"customer_id": cid})
        if "arrival" in row and not pd.isna(row["arrival"]):
            rec["arrival"] = row["arrival"]
        if "basket" in row and not pd.isna(row["basket"]):
            rec["basket"] = row["basket"]
        if "abandoned" in row and row.get("abandoned", False):
            rec["abandoned"] = True
            rec["abandon_time"] = row.get("abandon_time", rec.get("arrival", None))
        if "start_service" in row and not pd.isna(row["start_service"]):
            rec["start"] = row["start_service"]
            rec["assigned_server"] = int(row.get("assigned_server", -1))
        if "end_service" in row and not pd.isna(row["end_service"]):
            rec["end"] = row["end_service"]
    # produce list
    return [records[k] for k in sorted(records.keys())]

def make_animation(timelines, sim_minutes, num_checkouts, fps=6, interval_sec=0.5):
    # frames every interval (minutes)
    dt = 1.0 / fps * (interval_sec)  # not used; instead choose frame_time_step minutes
    frame_step = 0.5  # minutes per frame
    times = np.arange(0, sim_minutes + frame_step, frame_step)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 30)  # x-axis is schematic coordinate; we'll map queue length to left
    ax.set_ylim(-1, num_checkouts + 1)
    ax.axis('off')
    server_x = 22
    server_y = np.arange(num_checkouts)

    scat = ax.scatter([], [], s=80)

    # draw server boxes
    for i, y in enumerate(server_y):
        ax.add_patch(plt.Rectangle((server_x-1.5, y-0.35), 3.0, 0.7, fill=False))
        ax.text(server_x+1.8, y, f"S{i}", va="center")

    def get_positions_at(t):
        # for each customer, determine x,y
        xs, ys, cs = [], [], []
        for rec in timelines:
            arrival = rec.get("arrival", None)
            if arrival is None or arrival > t:
                continue
            if rec.get("abandoned", False) and rec.get("abandon_time", 1e9) <= t:
                # show faded dot at left
                xs.append(0)
                ys.append(-0.7)
                cs.append('grey')
                continue
            start = rec.get("start", None)
            end = rec.get("end", None)
            server = rec.get("assigned_server", None)
            if start and start <= t and (not end or t < end):
                # in service -> place in server box
                xs.append(server_x)
                ys.append(server)
                cs.append('tab:blue')
            elif end and t >= end:
                # finished -> place off to the right
                xs.append(server_x + 4 + 0.2 * (t - end))
                ys.append(server)
                cs.append('tab:green')
            else:
                # waiting in queue: position depends on arrival order and time
                # compute queue index by counting others who have arrival <= t and start > t
                waiting = []
                for r2 in timelines:
                    a2 = r2.get("arrival", 1e9)
                    s2 = r2.get("start", 1e9)
                    if a2 <= t and (s2 is None or s2 > t) and not r2.get("abandoned", False):
                        waiting.append(r2)
                # find position in waiting list
                try:
                    pos = waiting.index(rec)
                except ValueError:
                    pos = 0
                # map pos to x coordinate (left side starts at x=2)
                x = 2 + pos * 1.0
                # choose y row by round-robin to distribute visually
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
st.session_state["queue_policy"] = queue_policy

num_checkouts = st.sidebar.slider("Number of checkouts", 1, 12, 4)
express_count = st.sidebar.slider("Number of express checkouts (first N)", 0, num_checkouts, 1)
express_max_items = st.sidebar.slider("Express max items", 1, 20, 8)
st.sidebar.write("---")
st.sidebar.write("Cashier speeds (multiplier)")
cashier_speeds = []
cols = st.sidebar.columns(2)
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
        # convert events into tidy table per customer
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
        st.metric("Abandoned customers", summary_df["abandoned"].sum())
        st.metric("Avg wait (min)", f"{avg_wait:.2f}")
        st.metric("Median wait (min)", f"{median_wait:.2f}")
        st.metric("Throughput (cust/hr)", f"{throughput:.1f}")

        st.download_button("Download events CSV", df.to_csv(index=False).encode('utf-8'),
                           file_name="events.csv", mime="text/csv")

        st.dataframe(summary_df.sort_values("arrival").head(200))

        # Animation (build and display)
        with st.spinner("Building animation..."):
            timelines_list = timelines
            gif_buf = make_animation(timelines_list, sim_minutes=sim_minutes, num_checkouts=num_checkouts)
            st.image(gif_buf.getvalue(), caption="Simulation animation (schematic)")

        # Small charts
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        # wait histogram
        waits = served["start"].sub(served["arrival"]).dropna()
        ax[0].hist(waits, bins=30)
        ax[0].set_title("Wait time distribution (served)")
        ax[0].set_xlabel("Minutes")

        # service time histogram
        services = served["end"].sub(served["start"]).dropna()
        ax[1].hist(services, bins=30)
        ax[1].set_title("Service time distribution (served)")
        ax[1].set_xlabel("Minutes")

        st.pyplot(fig)
