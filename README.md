

<!-- `python experiment_tenn2.py train --processes 16 --sim_time 1000 --agents 10 --training_network networks/experiment_tenn2_mill36w23_1000t_n10.json ` -->
python experiment_tenn2.py train -N 10 --sim_time 1000 --processes 24 --network networks/experiment_tenn2_mill2023w37_1000t_n10_p1000_s20.json --pop 100 --epochs 100 --eons_seed 20 --graph_distribution 1000t_n10_p100_s20.tsv

python experiment_tenn2.py train -N 10 --sim_time 1000 --processes 24 --network networks/experiment_tenn2_mill20240215_0958_1000t_n10_p1000_s20.json --pop 100 --epochs 9999 --eons_seed 20 --graph_distribution 20240215_1000t_n10_p100_s20.tsv

python experiment_tenn2.py test --sim_time 1000 --agents 10 --network networks/experiment_tenn1_gs34w23_3000t_n20.json --all_counts_stream '{"source":"serve","port":8100}'

python experiment_tenn2.py run --sim_time 1000 --agents 10 --network results/p100/experiment_tenn2_mill2023w38_1000t_n10_p100_s20_e200.json --all_counts_stream '{"source":"serve","port":8100}'

python experiment_tenn3.py train --sim_time 1000 --agents 10 --network networks/experiment_tenn2_mill2023w38_1000t_n10_p100_s20_e100.json --pop 100 --epochs 100 --eons_seed 20 --graph_distribution zespol1000t_n10_p100_s20.tsv

rem visualizer command example
love . -n ../../turtwig/networks/experiment_tenn1_gs34w23_3000t_n20.json --show_input_id --show_output_id -i '{"source":"request","port":"8100", "host": "localhost"}'


love . -n ../../turtwig/results/p100/experiment_tenn2_mill2023w38_1000t_n10_p100_s20_e200.json --show_input_id --show_output_id --remove_unnecessary_neuron -i '{"source":"request","port":"8100", "host": "localhost"}'

python experiment_tenn2.py run --sim_time 1000 --agents 10 --network results/p100/experiment_tenn2_mill2023w38_1000t_n10_p100_s20_e200.json --all_counts_stream '{"source":"serve","port":8100}'