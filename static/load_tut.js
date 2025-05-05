function loadImage1(ref,imagepath) {
    const canvas = document.getElementById(ref)//'ref_img1');
    const ctx = canvas.getContext('2d');

    // Create an image object
    const img = new Image();
    // Draw the image on the canvas once it's loaded
  

    img.onload = function() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    img.src =imagepath;// 'img/ref1.jpg';
  }

  window.onload = function () {
    // Replace with your image path or URL
    loadImage1('ref_img1','img/ref1.jpg');
    loadImage1('ref_img2','img/ref2.jpg'); 
    introJs().start();

    setTimeout(poseImages, 500); // 2000 milliseconds = 2 seconds
    // startCamera();
     
 };