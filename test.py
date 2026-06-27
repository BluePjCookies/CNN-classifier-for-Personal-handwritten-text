from cnn import process, df_test, map_to_asci
from tensorflow import keras

model = keras.models.load_model(
    "model/emnist_cnn.keras"
)


map_dict = {}

with open(map_to_asci, "r") as file:
    
    for line in file:
        # Split each line at the first colon
        key, value = line.strip().split(" ", 1)
        # Store clean keys and values in the dictionary
        map_dict[key.strip()] = int(value.strip())

y_test, x_test = process(df_test)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(test_loss, test_accuracy) # 0.3629395067691803 0.8761104345321655