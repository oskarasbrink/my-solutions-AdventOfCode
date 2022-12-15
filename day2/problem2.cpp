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

// part two:
// X = loss
// y = draw
// z = win


int shape_score(std::string me){
    if(me == "X"){
        return 1;
    }else if (me == "Y"){
        return 2;
    }else{
        return 3;
    }


}

std::string strategy(std::string opp){
    if(opp=="A") return "X";
    else if (opp == "B") return "Y";
    else return "Z";
}
int outcome_score(std::string me, std::string opp){
    if(me == strategy(opp)){
        return 3 + shape_score(me);
    }
    //rock
    if(me == "X"){

        if(strategy(opp)=="Y") return shape_score(me);
        else if(strategy(opp)=="Z") return 6 + shape_score(me);
    }
    //paper
    if(me=="Y"){
        if(strategy(opp)=="X") return 6 + shape_score(me);
        else if (strategy(opp)=="Z") return shape_score(me);
    }
    //scissors
    if(me=="Z"){
        if(strategy(opp)=="X") return shape_score(me);
        else if(strategy(opp)=="Y") return 6 + shape_score(me);
    }
    return 999999;
}


int main(){

    std::ifstream i("day2Input.txt");
    std::string item;
    std::stringstream strStream;
    std::istringstream isstream;
    //isstream << i.rdbuf();
    strStream << i.rdbuf();
    i.close();

    std::string line,me,opp;
    int total_score = 0;
    while (std::getline(strStream,line,'\n')){
        opp = line[0];
        me = line.back();
        total_score += outcome_score(me,opp);
    }
    std::cout<<total_score<<std::endl;



    return 0;
}
