var audio = document.getElementById('audio');
var count=0;

document.body.onkeyup = function(e){
    if(count == 0){
        if (e.keyCode === 32 || e.key === ' ')
        {
            audio.play();
            count = 1;
        }
    }

}