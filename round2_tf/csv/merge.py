csv1 = 'stage_1_train_and_test_a.csv'
csv2 = 'fashionAI_key_points_test_b_answer_20180426.csv'


file1 = open(csv1,'a')
file2 = open(csv2,'r')
i=1
for line in file2:
    if i==1:
        i+=1
        continue
    file1.write(line)