import random

def get_choices():
    player_choice = input("Enter a choice(rock, paper, scissors): ")
    options = ["rock", "paper", "scissors"]
    computer_choice = random.choice(options)
    choices = {"player": player_choice, "computer": computer_choice}
    return choices
    return computer_choice


def check_win(player, computer):
    
    #print("you chose" + player + "computer chose" + computer)
    print(f"player chose {player} and computer chose {computer}")
    if player == "rock":
        if computer == "rock":
            print("it's a tie!")
        elif computer == "paper":
            print("you lose!")
        elif computer == "scissors":
            print("you win!")

    elif player == "paper":
        if computer == "rock":
            print("it's a win!")
        elif computer == "paper":
            print("you have tie!")
        elif computer == "scissors":
            print("you lose!")

    elif player == "scissors":
        if computer == "rock":
            print("it's a lose!")
        elif computer == "paper":
            print("you have win!")
        elif computer == "scissors":
            print("you have a tie!")



player = input("enter the choice(rock, paper, scissors: ")
options = ["rock", "paper", "scissors"]
computer = random.choice(options)

check_win(player, computer)






