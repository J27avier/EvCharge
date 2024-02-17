ps -aux | grep Run | awk '{print "kill", $2}' | bash
