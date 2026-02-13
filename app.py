import streamlit as st
import simpy, random, numpy as np, matplotlib.pyplot as plt

st.set_page_config(page_title="Retail Checkout Simulator", layout="wide")

# -------------------------
# Distribution Functions
# -------------------------
def sample_service_time(dist_name, mean, sigma):
    if dist_name == "exponential":
        return random.expovariate(1.0 / mean)
    elif dist_name == "normal":
        return max(0.01, random.gauss(mean, sigma))
    elif dist_name == "lognormal":
        return float(np.random.lognormal(mean=np.log(mean), sigma=sigma))
    elif dist_name == "fixed":
        return mean
    return random.expovariate(1.0 / mean)


# -------------------------
# Simulation Engine
# -------------------------
def run_simulation(sim_minutes, arrival_rate, num_checkouts,
                   service_dist, mean_service, sigma_service, seed):

    random.seed(seed)
    np.random.seed(seed)
    env = simpy.Environment()
    checkouts = simpy.Resource(env, capacity=num_checkouts)

    wait_times = []
    service_times = []
    queue_lengths = []
    times = []

    def monitor():
        while True:
            queue_lengths.append(len(checkouts.queue))
            times.append(env.now)
            yield env.timeout(0.5)

    def customer():
        arrival = env.now
        with checkouts.request() as req:
            yield req
            wait = env.now - arrival
            wait_times.append(wait)

            service_time = sample_service_time(service_dist, mean_service, sigma_service)
            service_times.append(service_time)
            yield env.timeout(service_time)

    def arrivals():
        while True:
            yield env.timeout(random.expovariate(arrival_rate/60))
            env.process(customer())

    env.process(arrivals())
    env.process(monitor())
    env.run(until=sim_minutes)

    return wait_times, service_times, times, queue_lengths


# -------------------------
# UI
# -------------------------
st.title("Retail Checkout Simulator")

with st.sidebar:
    sim_hours = st.slider("Simulation length (hours)", 1, 12, 4)
    arrival_rate = st.slider("Customers per hour", 10, 500, 200)
    num_checkouts = st.slider("Number of checkouts", 1, 12, 4)
    service_dist = st.selectbox("Service time distribution",
                                ["exponential", "normal", "lognormal", "fixed"])
    mean_service = st.number_input("Average service time (minutes)", 0.5, 10.0, 2.5)
    sigma_service = st.number_input("Sigma (for normal/lognormal)", 0.1, 5.0, 0.5)
    seed = st.number_input("Random seed", 1, 99999, 42)
    run_button = st.button("Run Simulation")

if run_button:
    wait_times, service_times, times, queue_lengths = run_simulation(
        sim_hours*60,
        arrival_rate,
        num_checkouts,
        service_dist,
        mean_service,
        sigma_service,
        seed
    )

    st.subheader("Results")

    st.write(f"Average wait time: {np.mean(wait_times):.2f} minutes")
    st.write(f"Max wait time: {np.max(wait_times):.2f} minutes")
    st.write(f"Customers processed: {len(wait_times)}")

    fig, ax = plt.subplots()
    ax.plot(times, queue_lengths)
    ax.set_title("Queue Length Over Time")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Queue Length")
    st.pyplot(fig)

    fig2, ax2 = plt.subplots()
    ax2.hist(wait_times, bins=30)
    ax2.set_title("Wait Time Distribution")
    st.pyplot(fig2)
