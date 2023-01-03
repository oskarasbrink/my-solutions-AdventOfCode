import scala.io.Source
import scala.collection.mutable.ListBuffer

@main def main() = {

    val filename: String = "day4Input.txt"
    //val lines = scala.io.Source.fromFile(filename).mkString
    val lines: List[String] = scala.io.Source.fromFile(filename).getLines.toList
    var numContainedRanges: Int = 0
    
    var numPartlyContainedRanges:Int = 0
    for(line <- lines){
        var splitLine = line.split(",")
        var n1:Int = splitLine(0).split("-")(0).toInt
        var n2:Int = splitLine(0).split("-")(1).toInt
        var n3:Int = splitLine(1).split("-")(0).toInt
        var n4:Int = splitLine(1).split("-")(1).toInt
        if (((n1 <= n3) && (n2 >= n4)) || (n3 <= n1 && n4 >= n2)){
            numContainedRanges += 1
        }
        
        //part two
        if (((n3 <= n2) && (n3 >= n1)) || (n1 <= n4 && n1 >= n3)){
            numPartlyContainedRanges += 1
        }

    }
    println(numContainedRanges)
    println(numPartlyContainedRanges)

    

    

   
}


