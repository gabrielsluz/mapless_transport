# for d in 30 25 20;
# do
#     for map_name in square_middle_corr_ four_squares_corr_ parallel_walls_corr_ mapless_1_corr_ mapless_2_corr_;
#     do
#         for obj_i in {0..9}
#         do
#             if [ $d = 20 ]; then
#                 python mapless_transport_eval.py $obj_i $map_name$d 8.5 10.0 100
#             elif [ $d = 25 ]; then
#                 python mapless_transport_eval.py $obj_i $map_name$d 11.5 11.5 100
#             elif [ $d = 30 ]; then
#                 python mapless_transport_eval.py $obj_i $map_name$d 12.0 12.0 100
#             fi
#         done
#     done
# done

d=20
for map_name in square_middle_corr_ four_squares_corr_ parallel_walls_corr_ mapless_1_corr_ mapless_2_corr_;
do
    for obj_i in {0..9}
    do
        python mapless_transport_eval.py $obj_i $map_name$d 9.0 9.0 100
    done
done
