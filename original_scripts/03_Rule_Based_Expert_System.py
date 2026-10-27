def expert_system():
    print("RULE-BASED EXPERT SYSTEM")

    fever = input("Do you have fever? (yes/no): ")
    cough = input("Do you have cough? (yes/no): ")
    cold = input("Do you have cold? (yes/no): ")

    if fever == "yes" and cough == "yes":
        print("Result: Flu")
    elif cold == "yes" and cough == "yes":
        print("Result: Common Cold")
    elif fever == "yes":
        print("Result: Infection")
    else:
        print("Result: No clear diagnosis")


expert_system()
