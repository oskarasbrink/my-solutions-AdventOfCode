//
//  problem6.cpp
//  AdventOfCode
//
//  Created by Oskar Åsbrink on 2022-12-10.
//

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <map>
#include <algorithm>
#include <numeric>
#include <string>
#include <utility>


bool check_dups(std::string s,int n_distinct_chars){
    
    for(int i = 0;i<n_distinct_chars;i++){
        if(std::count(s.begin(), s.end(), s[i])>1){
            return false;
        }
    }
    return true;
    
    
}
int main(){
        
    char c;

    std::ifstream i("/Users/oskarasbrink/Documents/GitHub/my-solutions-AdventOfCode/day6/day6Input.txt");
    std::map<char,int> m1;
    std::string sMAN,temp="";
    int n_distinct_chars = 14;
    int counter = 0;
    while(i>>c){
        sMAN +=c;
        
        if(sMAN.length()>n_distinct_chars-1){
            if(check_dups(sMAN.substr(sMAN.length()-n_distinct_chars,n_distinct_chars),n_distinct_chars)){
                break;
            }
        }
        //m1[c] += 1;
        
    }

                            
    
    std::cout<<sMAN.length()<<" yeah man";
    return 0;
}

