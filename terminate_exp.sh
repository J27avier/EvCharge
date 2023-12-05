ps -aux | grep RunCh | awk '{print "kill", $2}' | bash


