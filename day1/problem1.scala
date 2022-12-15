import scala.io.Source


@main def main() = {
    val filename: String = "input1.txt"
    //val lines = scala.io.Source.fromFile(filename).mkString
    //println(lines)
    val lines: List[String] = scala.io.Source.fromFile(filename).getLines.toList
    var value: Int = 0
    var max: Int = 0
    var elves: Vector[Int] = Vector()
   // println(lines)
    for(i <- lines) {
        //println(i)
        if (i == ""){
            if (value > max) {
                println("tjohej")
                max = value;
                
            }
            elves = elves :+ value
            value = 0
        } else {
            value += i.toInt;
            
        }
    }
    println(max)
    println(elves.sorted.reverse.take(3).sum)
}


