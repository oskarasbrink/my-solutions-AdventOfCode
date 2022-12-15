//
//  main.cpp
//  AdventOfCode
//
//  Created by Oskar Åsbrink on 2022-12-07.
//

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <vector> //for part two
#include <numeric>

int main() {

    std::ifstream i("input1.txt");
    std::string item;
    std::stringstream strStream;
    strStream << i.rdbuf();
    i.close();
    std::string str = strStream.str();
    std::string line;
    std::vector<int> elves;
    int sum = 0;
    int curr = 0;
    while (std::getline(strStream,line,'\n')){

        if (line.empty()){
            elves.push_back(curr);
            if (curr > sum){
                sum = curr;
            }
            curr = 0;

        }else{
        curr += stoi(line);
        }

    }
    sort(elves.begin(), elves.end(), std::greater<int>());
    int sum_of_top3_elves = std::accumulate(elves.begin(), elves.begin()+3,
                                   decltype(elves)::value_type(0));

    std::cout<<sum<<'\n';
    std::cout<<sum_of_top3_elves<<'\n';


}
