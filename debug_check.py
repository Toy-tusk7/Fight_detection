print("TEST START")

try:
    from train_lstm import ViolenceLSTM
    print("SUCCESS: ViolenceLSTM imported!")

    # Try to create model
    model = ViolenceLSTM()
    print("SUCCESS: Model instantiated!")

except Exception as e:
    print("ERROR DURING IMPORT:")
    print(e)

print("TEST END")
