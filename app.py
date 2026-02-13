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
        s
