//
//  problem3.cpp
//  AdventOfCode
//
//  Created by Oskar Åsbrink on 2022-12-07.
//

#include "problem3.hpp"
#include <iostream>
#include <cctype>
#include <string>
#include <algorithm>
#include <fstream>
#include <sstream>
int get_priority(char letter){
    //get ascii value
    // A has priority value 27
    // a har priority value 1
    // get ascii number and subtract
    if(isupper(letter)){
        return (int) letter - 38;
        }
    else{
        return (int) letter - 96;
    }
}

std::string get_intersection(std::string s1, std::string s2){
    std::string intersection = "";

    for(int i=0;i<s1.length();i++){
        if(s1.find(s2[i])<s1.length()){
            //check if already in substring
            if (intersection.find(s2[i])>intersection.length()){
                intersection += s2[i];
            }

        }
    }
    return intersection;
}

int calc_priority(std::string intersection){
    int score = 0;
    for(int i=0;i<intersection.length();i++){
        score += get_priority(intersection[i]);
    }
    return score;
}

int main(){

    std::ifstream i("day3Input.txt");
    std::string item;
    std::stringstream strStream;
    std::istringstream isstream;
    //isstream << i.rdbuf();
    strStream << i.rdbuf();
    i.close();
    std::string string_intersection;
    std::string line, first_half,last_half;
    char c;
    int score = 0;
    while (std::getline(strStream,line,'\n')){
        first_half = line.substr(0,line.length()/2);
        last_half = line.substr(line.length()/2,line.length());

        string_intersection = get_intersection(first_half, last_half);
        score += calc_priority(string_intersection);
    }
    std::cout<<score;





    return 0;
}
