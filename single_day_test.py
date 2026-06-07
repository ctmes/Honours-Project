from gymnax_exchange.jaxlobster.lobster_loader import LoadLOBSTER_resample
loader = LoadLOBSTER_resample(
    datapath="data", atpath=".",
    stock="AMZN", time_period="2022",
    n_Levels=10, type_="fixed_time",
    window_length=1800, window_resolution=60,
    n_data_msg_per_step=100
)
msgs, starts, ends, obs, max_msgs = loader.run_loading("test_1day")
print(msgs.shape, starts.shape)
