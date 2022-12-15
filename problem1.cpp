#include <iostream>
#include <vector>
#include <fstream>



int main() {

    std::ifstream instream("input1.txt");
    if (instream.fail()) {
        std::cerr << "Input file opening failed." << std::endl;
        exit(1);
    }
    while (instream) {
        std::string strInput;
        instream >> strInput;
        std::cout << strInput << std::endl;
    }


    return EXIT_SUCCESS;
}

