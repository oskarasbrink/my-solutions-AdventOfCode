import scala.io.Source
import scala.collection.mutable.ListBuffer
import scala.util.control.Breaks._

@main def main() = {

    val filename: String = "day6Input.txt"
    val lines: String = scala.io.Source.fromFile(filename).mkString
    breakable{
    for(n <- 4 to lines.length-1) {
        if(lines.substring(n-4,n).toCharArray.distinct.length>3){
            println(lines.substring(n-4,n))
            println(n)
            break
        } 

    }
    }
    //part two
    breakable{
    for(n <- 14 to lines.length-1) {
        if(lines.substring(n-14,n).toCharArray.distinct.length>13){
            println(lines.substring(n-14,n))
            println(n)
            break
        } 

    }
    }

}