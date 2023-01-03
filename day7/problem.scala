import scala.io.Source
import scala.collection.mutable.ListBuffer
import scala.util.control.Breaks._
//lets do some oop!
// why didnt i do stack instead of recursion......
//........
class dir(var s:String = " "):
    var size: Int = 0
    val dirname: String = s
    var subdirs: ListBuffer[dir] = ListBuffer()
    var filesInDir: ListBuffer[Int] = ListBuffer()
    var bestSum: Int = 0
    var bestList: ListBuffer[Int] = ListBuffer()
    var parent: dir = null


    def addSubDir(d:dir): Unit = {
        subdirs += d
    }
    def addFile(n:Int): Unit = {
        filesInDir += n
    }
    //recurse this badboy?
    def dirExists(c:String): Boolean = {
        if(subdirs.isEmpty){
            //println(s"found no subdir for dir $c")
            return false
        }
        for(dir<- this.subdirs){
            if(dir.dirname == c){
                //println(s"found subdir for dir $c")
                return true
            }else{
            //println(s"found no subdir $c, recursing some more")
            //println(s"parents parent is ${dir.parent.parent.dirname}")
            return dir.dirExists(c)
            }
        }
        //scala....
        return false
    }
    def getImmediateDirSize(): Int = {
        return filesInDir.sum
    }
    // recurse another badboy?
    def calcSizes(): Int = {
        if(this.subdirs.isEmpty){
            size = this.getImmediateDirSize()
            //println(s"immediate dir size is $size for ${this.dirname}")
            return size
        }else{
        var sum: Int = 0
        
        //println(s"current subdirs for ${this.dirname} is ${this.subdirs}")
        size = this.getImmediateDirSize()
        for(subdir<-this.subdirs){
            //println(s"subdir ${subdir.dirname} is $subdir")
            //println(s"recursing for ${subdir.dirname}")
            size += subdir.calcSizes()            
        }    
        }
        return size
    }
    def printSizes(): Unit = {
        //println(s"${this.dirname} has size $size")
        if(this != getTopLevelParent())
            getTopLevelParent().bestSum += size
            getTopLevelParent().bestList += size
        for(subdir<-subdirs){
            subdir.printSizes()
        }
    }
    def returnRightSizeDirs():Int = {

        if(!this.subdirs.isEmpty){
            var sum:Int = 0
            for(dir<-subdirs){
                sum += dir.returnRightSizeDirs()
            }
        }
        if(this.subdirs.isEmpty){//} && size <= 100000){
            return size
        }
        else{

         return 0}
    }
    def getTopLevelParent(): dir = {
        if(parent!=null){
            return parent.getTopLevelParent()
        }
        else{
            return this
        }
    }
end dir
@main def main() = {
    
    val lines = scala.io.Source.fromFile("input.txt").getLines.toList
    var parent = new dir("/")
    var currentDir = parent
    var ogParent = parent   
    for(line<-lines.tail){
        //println(line)
        //println(s"line: ${line}")
        if(line contains ".."){
                currentDir = parent
                if(parent.dirname!="/"){
                    parent = parent.parent
                }
                //println(s"parent is now ${parent.dirname}")
                //println(s"current dir is now ${currentDir.dirname}")
                
        }
        else if(line.take(4)=="$ cd"){
            val dirname = line.split(" ")(2)
            if(!currentDir.dirExists(dirname)){
                
                var tempdir:dir = new dir(dirname)

                parent.addSubDir(tempdir)
                tempdir.parent = parent
                parent = tempdir
                //println(s"adding subdir ${tempdir.dirname} with code ${tempdir} with parent ${tempdir.parent.dirname}")
                currentDir = tempdir

            }
        }else if(line.charAt(0).isDigit){
            //println(s"adding file for current dir ${currentDir.dirname} with size ${line.split(" ")(0).toInt}")
            currentDir.addFile(line.split(" ")(0).toInt)
        }
    }
    
    



    //part two
    
    var whatever: Int = ogParent.calcSizes()    
    ogParent.printSizes()
    var availableSpace:Int = 70000000
    var neededSpace:Int = 30000000
    var max:Int = 0
    ogParent.bestList += whatever
    var best_list:ListBuffer[Int] = ogParent.bestList.sorted.reverse
    println(best_list)
    var unusedSpace:Int = availableSpace - whatever
    breakable{
    for(dir<-0 to best_list.length-1){
        
        if(unusedSpace + best_list(dir) < neededSpace){

            max = best_list(dir-1)
            break
        }
        }
    }
    print(max)
    
}