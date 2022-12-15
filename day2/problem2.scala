import scala.io.Source

def handPoints(hand:Char): Int = {
    if (hand == 'X') 1
    else if (hand == 'Y') 2
    else if (hand == 'Z') 3
    else -1
}

def points(opp:Char,player:Char): Int = {
    if (opp == player) 3
    else if (opp == 'A' && player == 'Y') 6
    else if (opp == 'A' && player == 'Z') 0
    else if (opp == 'A' && player == 'X') 3
    else if (opp == 'B' && player == 'X') 0
    else if (opp == 'B' && player == 'Z') 6
    else if (opp == 'B' && player == 'Y') 3
    else if (opp == 'C' && player == 'X') 6
    else if (opp == 'C' && player == 'Y') 0
    else if (opp == 'C' && player == 'Z') 3
    else -1
    
}

@main def main() = {

    val filename: String = "day2Input.txt"
    //val lines = scala.io.Source.fromFile(filename).mkString
    //println(lines)
    val lines: List[String] = scala.io.Source.fromFile(filename).getLines.toList
    var value: Int = 0  
    val mapIm = Map("AX" -> 'Z', 
                        "AZ" -> 'Y',
                        "AY" -> 'X',
                        "BX" -> 'X',
                        "BY" -> 'Y',
                        "BZ" -> 'Z',
                        "CX" -> 'Y',
                        "CY" -> 'Z',
                        "CZ" -> 'X'
                        )

    // X for rock
    // Y for paper
    // Z for scissors

    //A for Rock
    //B for paper
    //C for scissors
    var linespart2 = lines.map({line => points(line(0),mapIm(line.replace(" ", ""))) + handPoints(mapIm(line.replace(" ", "")))})
    var lines2 = lines.map({line => points(line(0),line(2)) + handPoints(line(2))})
    println(lines2.sum)
    println(linespart2.sum)

    

   
}


