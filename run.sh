pwd=`pwd`
docker run -it --rm -v $pwd:/work -p 5000:15000 yj0604park/serving bash
