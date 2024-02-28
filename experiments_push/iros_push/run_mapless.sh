python mapless_transport_eval.py 8 mapless_2_corr_30 12.0 12.0

for map_name in square_middle_corr_25 four_squares_corr_25 parallel_walls_corr_25 mapless_1_corr_25 mapless_2_corr_25;
do
    for obj_i in {0..9}
    do
        python mapless_transport_eval.py $obj_i $map_name 11.5 11.5
    done
done

for map_name in square_middle_corr_20 four_squares_corr_20 parallel_walls_corr_20 mapless_1_corr_20 mapless_2_corr_20;
do
    for obj_i in {0..9}
    do
        python mapless_transport_eval.py $obj_i $map_name 8.5 10.0
    done
done