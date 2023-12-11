function setup() {
    createCanvas(800, 800);
  }
  
  let rhythm = [1,1,1,0,1,0,1,0];
  let linepointsx =[];
  let linepointsy = [];
  function draw() {
    background(102);
    push();
    translate(width * 0.5, height * 0.5);
    let coordinates = polygon(0, 0, 200, 32);
    for (let i = 0; i < rhythm.length; i++)
      {
        drawOnsets(coordinates[0][((16+(i*4))%32)],coordinates[1][((16+(i*4))%32)],8,rhythm[i]);
      }
    for (let pts = 0; pts < linepointsx.length-1; pts++){
        let v0 = createVector(linepointsx[pts],linepointsy[pts]); 
        let v1 = createVector(linepointsx[pts+1],linepointsy[pts+1]);
        //drawArrow(v0, v1, 'red');
        line(linepointsx[pts],linepointsy[pts],linepointsx[pts+1],linepointsy[pts+1] );
    }
    pop();
  
  
  }
  
  function polygon(x, y, radius, npoints) {
    let angle = TWO_PI / npoints;
    let pointsx =[];
    let pointsy = [];
    beginShape();
    for (let a = 0; a < TWO_PI; a += angle) {
      let sx = x + cos(a) * radius;
      let sy = y + sin(a) * radius;
      pointsx.push(sx);
      pointsy.push(sy);
      vertex(sx, sy);
    }
    endShape(CLOSE);
    return [pointsx, pointsy];
  }
  function drawOnsets(x,y,radius,rhythm)
  {
    if(rhythm ==1){
      fill(52);
      circle(x, y, radius);
        linepointsx.push(x);
        linepointsy.push(y);
    }else{
      fill('white');
      circle(x,y,radius);
    }
  }
  
  