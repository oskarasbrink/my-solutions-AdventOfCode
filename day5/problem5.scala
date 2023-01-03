import scala.io.Source
import scala.collection.mutable.ListBuffer
import scala.util.control.Breaks._

def initNestedList(s:String): ListBuffer[ListBuffer[Char]] = {
    var nestedList = ListBuffer[ListBuffer[Char]]()
    for(i <- 0 to (s.length/4)){
        var list = ListBuffer[Char]()
        nestedList += list
    }
    return nestedList
}

@main def main() = {

    val filename: String = "day5Input.txt"
    val lines: List[String] = scala.io.Source.fromFile(filename).getLines.toList
    val nestedList = initNestedList(lines(0))
    breakable{
    for(line <- lines){

        if(line == "" || line(1) == '1'){
            // do something nice here. or not. placeholder.
        }else if(!(line contains "from")){
            for(i <- 0 to ((line.length+1)/4 - 1)){
                if(!line(1 + 4*i).isWhitespace){
                nestedList(i) +:= line(1 + 4*i)
                }
            }
        }else{
            var s1: Array[String] = line.split(" ")
            var from:Int = s1(3).toInt
            var to:Int = s1(5).toInt
            var n:Int = s1(1).toInt
            nestedList(to-1) = nestedList(to-1) ++ nestedList(from-1).takeRight(n)
            // for part two
            // nestedList(to-1) = nestedList(to-1) ++ nestedList(from-1).takeRight(n).reverse
            nestedList(from-1) = nestedList(from-1).dropRight(n)
           
        }
    }
    val lastChars = nestedList.map(x => x.last)
    println(lastChars.mkString)
}


}