import subprocess

def main():
    while True:
        print("\n\nMenu:")
        print("1. Generate Data")
        print("2. Train models, fast mode (a few hours but very un-accurate)")
        print("3. Train models, generational mode (a few days and potentially very accurate)")
        print("4. Train NN model (not yet implemented)")
        print("5. Use local models")
        print("6. Use Azure model")
        print("7. Exit")
        choice = input("Enter your choice: ")
        print("\n \n")

        if choice == '1':
            print("Generating data...")
            subprocess.run(["python", "data_generation.py"])
        elif choice == '2':
            print("Training model (fast)...")
            subprocess.run(["python", "fast_ai_trainer.py"])
        elif choice == '3':
            print("Training model (generational)...")
            subprocess.run(["python", "generational_ai_trainer.py"])
        elif choice == '4':
            print("Training NN model...")
            print("Not yet implemented.")
        elif choice == '5':
            print("Using local models...")
            subprocess.run(["python", "use_model.py"])
        elif choice == '6':
            print("Using Azure model...")
            subprocess.run(["python", "use_azure_model.py"])
        elif choice == '7':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()