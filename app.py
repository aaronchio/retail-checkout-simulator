# app.py
import streamlit as st
import simpy
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import math
import tempfile
import os
from io import BytesIO

st.set_page_config(page_title="Retail Checkout Simulator", layout="wide")

# -------------------------
# Utility Functions
# -------------------------

def safe_float(x, default=float("inf")):
    try:
        if x is None:
            return default
        return float(x)
    except:
        return default

def sample_basket(mean, sigma):
    return max(1, int(round(np.random.normal(mean, sigma))))

def service_time(basket, scan_time, overhead=0.5):
    return overhead + basket * scan_time

def arrival_rate(base, pattern, t):
    if pattern == "flat":
        return base
    if pattern == "morning_peak":
        return base * (1 + 1.5 * math.exp(-0.5*((t-120)/60)**2))
    if pattern == "double_peak":
        return base * (
            1 +
            1.2*math.exp(-0.5*((t-90)/50)**2) +
            1.2*math.exp(-0.5*((t-300)/50)**2)
        )
    return base

# -------------------------
# Simulation Core
# -------------------------

class CheckoutSim:

    def __init__(self, minutes, base_rate, pattern,
                 checkouts, express_count, express_limit,
                 speeds, basket_mean, basket_sigma,
                 patience, seed, scan_time):

        random.seed(seed)
        np.random.seed(seed)

        self.env = simpy.Environment()
        self.minutes = minutes
        self.base_rate = base_rate
        self.pattern = pattern
        self.num_checkouts = checkouts
        self.express_count = express_count
        self.express_limit = express_limit
        self.speeds = speeds
        self.basket_mean = basket_mean
        self.basket_sigma = basket_sigma
        self.patience = patience
        self.scan_time = scan_time

        self.checkouts = [simpy.Resource(self.env, capacity=1)
                          for _ in range(checkouts)]
        self.shared_queue = simpy.Resource(self.env, capacity=checkouts)

        self.events = []

    def run(self):
        self.env.process(self.arrivals())
        self.env.run(until=self.minutes)
        return pd.DataFrame(self.events)

    def arrivals(self):
        cid = 0
        while True:
            rate = arrival_rate(self.base_rate, self.pattern, self.env.now)
            lam = max(1e-6, rate/60)
            yield self.env.timeout(random.expovariate(lam))
            cid += 1
            self.env.process(self.customer(cid))
            if self.env.now >= self.minutes:
                break

    def shortest_server(self):
        lens = [len(r.queue) + r.count for r in self.checkouts]
        return int(np.argmin(lens))

    def customer(self, cid):
        arrival = self.env.now
        basket = sample_basket(self.basket_mean, self.basket_sigma)

        express = self.express_count > 0 and basket <= self.express_limit
        policy = st.session_state.get("queue_policy", "single")

        if express:
            server_idx = min(self.shortest_server(), self.express_count - 1)
            req = self.checkouts[server_idx].request()
            yield req

        elif policy == "multiple_shortest":
            server_idx = self.shortest_server()
            req = self.checkouts[server_idx].request()
            yield req

        elif policy == "multiple_random":
            server_idx = random.randrange(self.num_checkouts)
            req = self.checkouts[server_idx].request()
            yield req

        else:  # single shared queue
            req = self.shared_queue.request()
            yield req
            server_idx = self.shortest_server()
            req2 = self.checkouts[server_idx].request()
            yield req2

        start = self.env.now

        stime = service_time(basket, self.scan_time)
        stime = stime / max(1e-6, self.speeds[server_idx])

        self.events.append({
            "customer_id": cid,
            "arrival": arrival,
            "start": start,
            "server": server_idx,
            "basket": basket
        })

        yield self.env.timeout(stime)

        end = self.env.now

        self.events.append({
            "customer_id": cid,
            "end": end,
            "server": server_idx
        })

        self.checkouts[server_idx].release(
            req2 if policy == "single" and not express else req
        )
        if policy == "single" and not express:
            self.shared_queue.release(req)

# -------------------------
# Animation
# -------------------------

def build_timelines(df):
    recs = {}
    for _, row in df.iterrows():
        cid = row["customer_id"]
        if cid not in recs:
            recs[cid] = {}
        for col in row.index:
            recs[cid][col] = row[col]
    return list(recs.values())

def make_animation(timelines, minutes, checkouts):

    frame_step = 2.0  # fewer frames for reliability
    times = np.arange(0, minutes + frame_step, frame_step)

    fig, ax = plt.subplots(figsize=(8,4))

    def update(i):
        ax.clear()
        ax.set_xlim(0, 30)
        ax.set_ylim(-1, checkouts+1)
        ax.axis("off")

        t = times[i]

        waiting = [
            r for r in timelines
            if safe_float(r.get("arrival")) <= t
            and safe_float(r.get("start"), None) is None
        ]

        xs, ys, cs = [], [], []

        for r in timelines:
            arr = safe_float(r.get("arrival"))
            start = r.get("start")
            end = r.get("end")
            server = r.get("server", 0)

            if arr > t:
                continue

            if start and start <= t and (not end or t < end):
                xs.append(22); ys.append(server); cs.append("blue")
            elif end and t >= end:
                xs.append(26); ys.append(server); cs.append("green")
            else:
                pos = waiting.index(r) if r in waiting else 0
                xs.append(2 + pos)
                ys.append(pos % max(1, checkouts))
                cs.append("orange")

        ax.scatter(xs, ys, s=80, c=cs)
        ax.set_title(f"Minute {t:.1f}")

    anim = animation.FuncAnimation(fig, update, frames=len(times), repeat=False)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
    tmp_name = tmp.name
    tmp.close()

    anim.save(tmp_name, writer="pillow", fps=4)
    plt.close(fig)

    with open(tmp_name, "rb") as f:
        data = f.read()
    os.remove(tmp_name)

    return BytesIO(data)

# -------------------------
# UI
# -------------------------

st.sidebar.title("Simulation Controls")

sim_hours = st.sidebar.slider("Sim hours", 0.5, 6.0, 1.5)
base_rate = st.sidebar.slider("Customers per hour", 10, 400, 120)
pattern = st.sidebar.selectbox("Arrival pattern", ["flat","morning_peak","double_peak"])
queue_policy = st.sidebar.selectbox("Queue policy", ["single","multiple_shortest","multiple_random"])
st.session_state["queue_policy"] = queue_policy

checkouts = st.sidebar.slider("Checkouts", 1, 8, 4)
express_count = st.sidebar.slider("Express lanes", 0, checkouts, 1)
express_limit = st.sidebar.slider("Express max items", 1, 20, 8)

speeds = [st.sidebar.number_input(f"Speed {i+1}",0.2,3.0,1.0,0.1)
          for i in range(checkouts)]

basket_mean = st.sidebar.number_input("Avg basket size",1,100,12)
basket_sigma = st.sidebar.number_input("Basket sigma",0.0,50.0,3.0)
scan_time = st.sidebar.number_input("Scan time per item",0.05,1.0,0.25,0.01)
seed = st.sidebar.number_input("Seed",0,999999,42)

run = st.sidebar.button("Run Simulation")

st.title("Retail Checkout Simulator")

if run:
    sim = CheckoutSim(
        minutes=int(sim_hours*60),
        base_rate=base_rate,
        pattern=pattern,
        checkouts=checkouts,
        express_count=express_count,
        express_limit=express_limit,
        speeds=speeds,
        basket_mean=basket_mean,
        basket_sigma=basket_sigma,
        patience=None,
        seed=seed,
        scan_time=scan_time
    )

    df = sim.run()

    if not df.empty:
        timelines = build_timelines(df)
        gif = make_animation(timelines, int(sim_hours*60), checkouts)
        st.image(gif.getvalue(), caption="Checkout Animation")
        st.dataframe(df.head(200))
    else:
        st.write("No customers processed.")
