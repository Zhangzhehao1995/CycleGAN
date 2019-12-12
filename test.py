import tensorflow as tf

Dataset = tf.data.Dataset
a = Dataset.from_tensor_slices([1,2,3])
b = Dataset.from_tensor_slices([4,5,6])
c = Dataset.from_tensor_slices([(7,8),(9,10),(11,12),(13,14)])

dataset1 = Dataset.zip((a,b))
dataset2 = Dataset.zip((a,b,c))


for one_element in dataset1:
    print(one_element)

for one_element in dataset2:
    print(one_element)