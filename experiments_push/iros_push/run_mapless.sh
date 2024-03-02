for d in 20 25 30;
do
    for map_name in square_middle_corr_ four_squares_corr_ parallel_walls_corr_ mapless_1_corr_ mapless_2_corr_;
    do
        for obj_i in {0..9}
        do
            if [ $d = 20 ]; then
                python mapless_transport_eval.py $obj_i $map_name$d 8.5 10.0 100
            elif [ $d = 25 ]; then
                python mapless_transport_eval.py $obj_i $map_name$d 11.5 11.5 100
            elif [ $d = 30 ]; then
                python mapless_transport_eval.py $obj_i $map_name$d 12.0 12.0 100
            fi
        done
    done
done

# for map_name in parallel_walls_corr_25 mapless_2_corr_25;
# do
#     for obj_i in {0..9}
#     do
#         for mem_len in 100 200 300
#         do
#             python mapless_transport_eval.py $obj_i $map_name 11.5 11.5 $mem_len
#         done
#     done
# done

# for map_name in square_middle_corr_25 four_squares_corr_25 parallel_walls_corr_25 mapless_1_corr_25 mapless_2_corr_25;
# do
#     for obj_i in {0..9}
#     do
#         python mapless_transport_eval.py $obj_i $map_name 11.5 11.5
#     done
# done

# for map_name in square_middle_corr_20 four_squares_corr_20 parallel_walls_corr_20 mapless_1_corr_20 mapless_2_corr_20;
# do
#     for obj_i in {0..9}
#     do
#         python mapless_transport_eval.py $obj_i $map_name 8.5 10.0
#     done
# done