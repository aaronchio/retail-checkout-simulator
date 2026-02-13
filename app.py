# app.py
import streamlit as st
import simpy
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from io import BytesIO
import pandas as pd
import math
import tempfile
import os

st.set_page_config(page_title="Retail Checkout Simulator (Upgraded)", layout="wide")

# -------------------------
# Utility Functions
# -------------------------
def sample_basket_size(mean, sigma):
    return max(1, int(round(np.random.normal(mean, sigma))))

def sample_service_time_from_basket(basket_size, scan_time=0.3, base_overhead=0.5):
    return base_overhead + basket_size * scan_time

def time_varying_rate(base_rate, pattern, t):
    if pattern == "flat":
        return base_rate
    if pattern == "morning_peak":
        mu = 120
        sigma = 60
        factor = 1 + 1.5 * math.exp(-0.5 * ((t - mu) / sigma) ** 2)
        return base_rate * factor
    if pattern == "double_peak":
        mu1, mu2 = 90, 300
        sigma = 50
        factor = 1 + 1.2*math.exp(-0.5*((t-mu1)/sigma)**2) + 1.2*math.exp(-0.5*((t-mu2)/sigma)**2)
        return base_rate * factor
    return base_rate

# -------------------------
# Simulation Core
# -------------------------
class CheckoutSim:
    def __init__(self, sim_minutes, base_rate, arrival_pattern,
                 num_checkouts, express_count, express_max_items,
                 cashier_speeds, basket_mean, basket_sigma,
                 patience, seed, scan_time):

        self.sim_minutes = sim_minutes
        self.base_rate = base_rate
        self.arrival_pattern = arrival_pattern
        self.num_checkouts = num_checkouts
        self.express_count = express_count
        self.express_max_items = express_max_items
        self.cashier_speeds = cashier_speeds
        self.basket_mean = basket_mean
        self.basket_sigma = basket_sigma
        self.patience = patience
        self.scan_time = scan_time

        random.seed(seed)
        np.random.seed(seed)

        self.env = simpy.Environment()
        self.checkouts = [simpy.Resource(self.env, capacity=1) for _ in range(num_checkouts)]
        self.shared_queue = simpy.Resource(self.env, capacity=num_checkouts)

        self.events = []

    def run(self):
        self.env.process(self.arrivals())
        self.env.run(until=self.sim_minutes)
        return pd.DataFrame(self.events)

    def arrivals(self):
        cid = 0
        while True:
            rate = time_varying_rate(self.base_rate, self.arrival_pattern, self.env.now)
            lam = max(1e-6, rate / 60)
            yield self.env.timeout(random.expovariate(lam))
            cid += 1
            self.env.process(self.customer(cid))

            if self.env.now >= self.sim_minutes:
                break

    def choose_server_shortest(self):
        lens = [len(r.queue) + r.count for r in self.checkouts]
        return int(np.argmin(lens))

    def customer(self, cid):
        arrival = self.env.now
        basket = sample_basket_size(self.basket_mean, self.basket_sigma)
        is_express = self.express_count > 0 and basket <= self.express_max_items
        policy = st.session_state.get("queue_policy", "single")

        if is_express:
            idx = min(self.choose_server_shortest(), self.express_count - 1)
        elif policy == "multiple_shortest":
            idx = self.choose_server_shortest()
        elif policy == "multiple_random":
            idx = random.randrange(self.num_checkouts)
        else:
            idx = None

        if idx is None:
            req = self.shared_queue.request()
            if self.patience:
                result = yield req | self.env.timeout(self.patience)
                if req not in result:
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

            server_idx = self.choose_server_shortest()
            serv_req = self.checkouts[server_idx].request()
            yield serv_req
        else:
            req = self.checkouts[idx].request()
            if self.patience:
                result = yield req | self.env.timeout(self.patience)
                if req not in result:
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

        start = self.env.now
        base_service = sample_service_time_from_basket(basket, self.scan_time)
        service_time = base_service / max(1e-6, self.cashier_speeds[server_idx])

        self.events.append({
            "customer_id": cid,
            "arrival": arrival,
            "basket": basket,
            "assigned_server": server_idx,
            "start_service": start,
            "abandoned": False
        })

        yield self.env.timeout(service_time)

        end = self.env.now

        self.events.append({
            "customer_id": cid,
            "end_service": end,
            "assigned_server": server_idx
        })

        self.checkouts[server_idx].release(req if idx is not None else serv_req)
        if idx is None:
            self.shared_queue.release(req)

# -------------------------
# Animation
# -------------------------
def build_timelines(df):
    records = {}
    for _, row in df.iterrows():
        cid = row.get("customer_id")
        if cid not in records:
            records[cid] = {"customer_id": cid}
        if "arrival" in row and not pd.isna(row.get("arrival")):
            records[cid]["arrival"] = row.get("arrival")
        if "start_service" in row:
            records[cid]["start"] = row.get("start_service")
            records[cid]["server"] = row.get("assigned_server")
        if "end_service" in row:
            records[cid]["end"] = row.get("end_service")
        if row.get("abandoned", False):
            records[cid]["abandoned"] = True
            records[cid]["abandon_time"] = row.get("abandon_time")
    return list(records.values())

def make_animation(timelines, sim_minutes, num_checkouts):

    frame_step = 0.5
    times = np.arange(0, sim_minutes + frame_step, frame_step)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.set_xlim(0, 30)
    ax.set_ylim(-1, num_checkouts + 1)
    ax.axis('off')

    server_x = 22

    def get_positions(t):
        xs, ys, cs = [], [], []
        waiting = [r for r in timelines
                   if r.get("arrival", 1e9) <= t and
                   not r.get("abandoned", False) and
                   (r.get("start") is None or r.get("start") > t)]

        for r in timelines:
            arrival = r.get("arrival", 1e9)
            if arrival > t:
                continue

            if r.get("abandoned", False) and r.get("abandon_time", 1e9) <= t:
                xs.append(0); ys.append(-0.5); cs.append("gray")
                continue

            start = r.get("start")
            end = r.get("end")
            server = r.get("server")

            if start and start <= t and (not end or t < end):
                xs.append(server_x); ys.append(server); cs.append("blue")
            elif end and t >= end:
                xs.append(server_x + 4); ys.append(server); cs.append("green")
            else:
                pos = waiting.index(r) if r in waiting else 0
                xs.append(2 + pos); ys.append(pos % num_checkouts); cs.append("orange")

        return xs, ys, cs

    def update(i):
        ax.clear()
        ax.set_xlim(0, 30)
        ax.set_ylim(-1, num_checkouts + 1)
        ax.axis('off')
        t = times[i]
        xs, ys, cs = get_positions(t)
        ax.scatter(xs, ys, s=80, c=cs)
        ax.set_title(f"Minute {t:.1f}")

    anim = animation.FuncAnimation(fig, update, frames=len(times), repeat=False)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
    tmp_name = tmp.name
    tmp.close()

    anim.save(tmp_name, writer="pillow", fps=6)
    plt.close(fig)

    with open(tmp_name, "rb") as f:
        data = f.read()

    os.remove(tmp_name)
    return BytesIO(data)

# -------------------------
# Streamlit UI
# -------------------------
st.sidebar.title("Simulation Controls")

sim_hours = st.sidebar.slider("Sim length (hours)", 0.5, 8.0, 2.0)
base_rate = st.sidebar.slider("Arrival rate (cust/hr)", 10, 500, 150)
arrival_pattern = st.sidebar.selectbox("Arrival pattern", ["flat","morning_peak","double_peak"])
queue_policy = st.sidebar.selectbox("Queue policy", ["single","multiple_shortest","multiple_random"])
st.session_state["queue_policy"] = queue_policy

num_checkouts = st.sidebar.slider("Checkouts", 1, 10, 4)
express_count = st.sidebar.slider("Express checkouts", 0, num_checkouts, 1)
express_max_items = st.sidebar.slider("Express max items", 1, 20, 8)

cashier_speeds = [st.sidebar.number_input(f"Speed {i+1}",0.2,3.0,1.0,0.1) for i in range(num_checkouts)]

basket_mean = st.sidebar.number_input("Avg basket size",1,100,12)
basket_sigma = st.sidebar.number_input("Basket sigma",0.0,50.0,3.0)
patience_input = st.sidebar.number_input("Patience (min, 0=none)",0.0,100.0,0.0)
patience = None if patience_input <= 0 else patience_input

scan_time = st.sidebar.number_input("Scan time per item",0.05,1.0,0.25,0.01)
seed = st.sidebar.number_input("Random seed",0,999999,42)

run_button = st.sidebar.button("Run Simulation")

st.title("Retail Checkout Simulator — Upgraded")

if run_button:
    with st.spinner("Running simulation..."):
        sim = CheckoutSim(
            sim_minutes=int(sim_hours*60),
            base_rate=base_rate,
            arrival_pattern=arrival_pattern,
            num_checkouts=num_checkouts,
            express_count=express_count,
            express_max_items=express_max_items,
            cashier_speeds=cashier_speeds,
            basket_mean=basket_mean,
            basket_sigma=basket_sigma,
            patience=patience,
            seed=seed,
            scan_time=scan_time
        )

        df = sim.run()

    if not df.empty:
        timelines = build_timelines(df)
        gif = make_animation(timelines, int(sim_hours*60), num_checkouts)
        st.image(gif.getvalue(), caption="Checkout Flow Animation")
        st.dataframe(df.head(200))
    else:
        st.write("No customers processed.")
