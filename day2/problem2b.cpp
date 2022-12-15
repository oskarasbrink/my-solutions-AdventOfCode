//
//  problem2.cpp
//  AdventOfCode
//
//  Created by Oskar Åsbrink on 2022-12-07.
//

#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <vector> //for part two
#include <numeric>

    // A = rock
    // B = paper
    // C = scissors

    // X = rock
    // Y = paper
    // Z = scissors

    /* Total score is the sum of scores for each round. The score for a single round
     is the score for the shape you selected (1 for rock, 2 for paper, 3 for scissors)
     plus the score of the outcome of the round (0 for loss, 3 for draw, 6 for win)

     ----Part two-----
    The Elf finishes helping with the tent and sneaks back over to you. "Anyway, the second column says how the round needs to end: X means you need to lose, Y means you need to end the round in a draw, and Z means you need to win. Good luck!"

    The total score is still calculated in the same way, but now you need to figure out what shape to choose so the round ends as indicated. The example above now goes like this:

    In the first round, your opponent will choose Rock (A), and you need the round to end in a draw (Y), so you also choose Rock. This gives you a score of 1 + 3 = 4.
    In the second round, your opponent will choose Paper (B), and you choose Rock so you lose (X) with a score of 1 + 0 = 1.
    In the third round, you will defeat your opponent's Scissors with Rock for a score of 1 + 6 = 7.
    Now that you're correctly decrypting the ultra top secret strategy guide, you would get a total score of 12.

    Following the Elf's instructions for the second column, what would your total score be if everything goes exactly according to your strategy guide?*/

int hand_score(std::string hand){
    if(hand == "ROCK"){
        return 1;
    }else if (hand == "PAPER"){
        return 2;
    }else{ //scissors
        return 3;
    }


}
int outcome_score(std::string outcome){
    if (outcome=="LOSS") return 0;
    else if (outcome == "DRAW") return 3;
    else return 6; //win
}




int round_score(std::string hand, std::string state){
    if(state == "DRAW") return hand_score(hand) + outcome_score(state);

    else if(state == "LOSS"){
        if (hand == "ROCK") return hand_score("SCISSORS") + outcome_score(state);
        if (hand == "PAPER") return hand_score("ROCK") + outcome_score(state);
        if (hand == "SCISSORS") return hand_score("PAPER") + outcome_score(state);
    }

    else if(state == "WIN"){
        if(hand == "ROCK") return hand_score("PAPER") + outcome_score(state);
        if(hand == "PAPER") return hand_score("SCISSORS") + outcome_score(state);
        if(hand == "SCISSORS") return hand_score("ROCK") + outcome_score(state);
    }else{
      std::cout<<"something fishy here...";
      return 9999; //only three scenarios...
    }
    std::cout<<"something fishy here...";
    return 999999;
}



std::string translate_hand(std::string hand){
    if(hand=="A") return "ROCK";
    else if (hand=="B") return "PAPER";
    else return "SCISSORS";
}

std::string translate_win(std::string state){
    if(state=="X") return "LOSS";
    else if(state=="Y") return "DRAW";
    else return "WIN";
}
int main(){
    std::ifstream i("day2Input.txt");
    std::string item;
    std::stringstream strStream;
    std::istringstream isstream;
    strStream << i.rdbuf();
    i.close();

    std::string line, hand, state;
    int total_score = 0;
    while (std::getline(strStream,line,'\n')){
        hand = line[0];
        state = line.back();
        std::cout<<hand<<" asd "<<state<<std::endl;
        total_score += round_score(translate_hand(hand),translate_win(state));
    }
    std::cout<<total_score<<std::endl;
    return 0;
}
