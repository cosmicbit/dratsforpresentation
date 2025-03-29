import streamlit as st
import pandas as pd
import numpy as np


from oneway.testing_simulation import test_agent as modelone
from oneway.ttl import run_simulation as ttlone
from twoway.test import test_agent as modeltwo
from twoway.ttl import run_simulation as ttltwo

st.set_page_config(page_title="DRATS", layout="wide")
st.title("Deep Reinforcement Approach on Traffic System Dashboard")

# Sidebar for simulation selection
model_choice = st.sidebar.radio("Select Model", ("One Intersection", "Two Intersections"))
st.sidebar.write("You selected:", model_choice)
if model_choice == "One Intersection":
    st.header("One Fourway Intersection")
    st.subheader("Training Rewards")
    st.image("oneway/images/Training_Rewards.jpg")
    if st.button("Test"):
        # subprocess.run(["python", "testing_simulation.py"])
        rewards, waiting_times = modelone()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Rewards over timesteps during testing")
            st.line_chart(pd.DataFrame({"Rewards over timesteps by RL agent" : rewards}))
        with col2:
            st.subheader("Waiting times over timesteps during testing")
            st.line_chart(pd.DataFrame({"Waiting times over timesteps by RL agent" :waiting_times}))
            st.write("Average waiting time by RL agent = ", np.mean(waiting_times))


        rewards, waiting_times = ttlone()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Rewards over timesteps from TTLS")
            st.line_chart(pd.DataFrame({"Rewards over timesteps from TTLS": rewards}))
        with col2:
            st.subheader("Waiting times over timesteps from TTLS")
            st.line_chart(pd.DataFrame({"Waiting times over timesteps from TTLS": waiting_times}))
            st.write("Average waiting time by TTLS = ", np.mean(waiting_times))

if model_choice == "Two Intersections":
    st.header("Two Fourway Intersections")
    st.subheader("Training Rewards")
    st.image("twoway/images/Training Rewards.jpg")
    if st.button("Test"):
        # subprocess.run(["python", "testing_simulation.py"])
        rewards, waiting_times = modeltwo()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Rewards over timesteps during testing")
            st.line_chart(pd.DataFrame({"Rewards over timesteps during testing": rewards}))
        with col2:
            st.subheader("Waiting times over timesteps during testing")
            st.line_chart(pd.DataFrame({"Waiting times over timesteps during testing": waiting_times}))

        rewards, waiting_times = ttltwo()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Rewards over timesteps from TTLS")
            st.line_chart(pd.DataFrame({"Rewards over timesteps from TTLS": rewards}))
        with col2:
            st.subheader("Waiting times over timesteps from TTLS")
            st.line_chart(pd.DataFrame({"Waiting times over timesteps from TTLS": waiting_times}))

