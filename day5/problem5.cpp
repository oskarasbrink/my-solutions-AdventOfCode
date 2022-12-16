//
//  problem5.cpp
//  AdventOfCode
//
//  Created by Oskar Åsbrink on 2022-12-10.
//

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <ios>
#include <list>
#include <algorithm>
using namespace std;


void process_line(std::string& line, std::vector<std::vector<char>>& bigboy,bool& init){
    
    
    std::string temp;
    
    std::vector<char> tempvector;
    for(int i = 0;i<(line.length()-3)/4+1;i++){
        std::cout<<line.length()<<" i: "<<i;
        temp = line.substr(4*i,3);
        
        if(!init){
            bigboy.push_back(tempvector);
        }
        if(temp != "   " && !isdigit(temp[1])){
            bigboy.at(i).insert(bigboy.at(i).begin(),temp[1]);
        }
        
        std::cout<<temp<<" "<<temp.length()<<std::endl;
    }
    init = true;
    
}

void init_stacks(std::stringstream& strStream,std::vector<std::vector<char>>& bigboy,bool& init){
    std::string line;
    
    
    while(std::getline(strStream,line,'\n')){
            
        //std::cout<<line<<std::endl;
        if(line==""){
            break;
        }
        
        process_line(line,bigboy,init);
    }
    std::cout<<"broke it";
    //print_bigman(bigboy);
}

void execute_instruction(std::vector<std::vector<char>>& bigman,std::string block,std::string from,std::string to){
    
    int f = stoi(from)-1;
    int b = stoi(block);
    int t = stoi(to)-1;
    std::vector<int> subvector = {bigman.at(f).end()-b,bigman.at(f).end()};
    //std::reverse(subvector.begin(),subvector.end());
    bigman.at(f).erase(bigman.at(f).end()-b,bigman.at(f).end());
    bigman.at(t).insert(bigman.at(t).end(),subvector.begin(),subvector.end());
    //print_bigman(bigman);
}

void print_result(std::vector<std::vector<char>>& bigman){
    std::cout<<std::endl;
    for(int i = 0;i<bigman.size();i++){
        std::cout<<bigman.at(i).back();
    }
    std::cout<<std::endl;
}
int main6(){
    
    std::ifstream i("/Users/oskarasbrink/Documents/GitHub/my-solutions-AdventOfCode/day5/day5Input.txt");
    std::string item;
    std::stringstream strStream;
    //isstream << i.rdbuf();
    strStream << i.rdbuf();
    i.close();

   
    std::vector<std::vector<char>> bigboy;
    std::string line;
    bool init = false;
    init_stacks(strStream,bigboy,init);
    
    std::string junk,block,from,to;
    while(std::getline(strStream,junk,' ')){
        std::getline(strStream,block,' ');
        std::getline(strStream,junk,' ');
        std::getline(strStream,from,' ');
        std::getline(strStream,junk,' ');
        std::getline(strStream,to,'\n');
        
        std::cout<<"asd "<<block[0]<<" "<<from[0]<<" "<<to[0]<<std::endl;
        //print_bigman(bigboy);
        execute_instruction(bigboy,block,from,to);
        //std::cout<<line<<std::endl;
        
        //move_line();
    }
    //print_bigman(bigboy);
    print_result(bigboy);
    
    
    
    return 0;
}
