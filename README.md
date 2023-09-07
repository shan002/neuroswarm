

python experiment_tenn2.py test --sim_time 1000 --agents 10 --network networks/experiment_tenn1_gs34w23_3000t_n20.json --all_counts_stream '{"source":"serve","port":8100}'

visualizer command example
`love . -n ../../turtwig/networks/experiment_tenn1_gs34w23_3000t_n20.json --show_input_id --show_output_id -i '{"source":"request","port":"8100", "host": "localhost"}'`