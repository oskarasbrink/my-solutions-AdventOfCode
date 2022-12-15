import scala.io.Source
import scala.collection.mutable.ListBuffer


def getPriorityValue(c:Char): Int = {
    if(c.isUpper){
        return c.toInt - 38
    }
    else{
        return c.toInt - 96
    }


}

@main def main() = {

    val filename: String = "day3Input.txt"
    //val lines = scala.io.Source.fromFile(filename).mkString
    val lines: List[String] = scala.io.Source.fromFile(filename).getLines.toList
    var intersection: ListBuffer[Int] = new ListBuffer[Int]()
    for(line <- lines) {
        var line1: String = line.take(line.length/2)
        var line2: String = line.substring(line.length/2)
        var intsect: String = line1 intersect line2
        intersection += getPriorityValue(intsect(0))
        
    }

    //part 2
    var intersectionP2: ListBuffer[Int] = new ListBuffer[Int]()
    for(i <- 0 until (lines.length)/3)
    {
        var line1: String = lines(3*i)
        var line2: String = lines(3*i+1)
        var line3: String = lines(3*i+2)
        var intersect: String = line1 intersect line2 intersect line3
        intersectionP2 += getPriorityValue(intersect(0))
    }
    println(intersectionP2.sum)
    

    

    

   
}


