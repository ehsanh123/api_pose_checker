

async function loadImage1(a) {
    const inputEl = document.getElementById("poseStatus");
    
    im_bax = 'ref_img1';
    inputEl.innerText = "Loading Refernce image 1...";
    if (a == '2') {
        im_bax = 'ref_img2';
        inputEl.innerText = "Loading referrance image 2";
    }

    const fileInput = document.getElementById("imageLoader");

    return new Promise((resolve) => {
        fileInput.onchange = function () {
            const file = fileInput.files[0];

            if (!file) {
                inputEl.innerText = "No file selected";
                resolve(null);
                return;
            }

            inputEl.innerText = "Refernce 1 loaded";
            if (a == '2') inputEl.innerText = "Refernce 3 loaded";
            

            // Optionally load image into canvas
            const reader = new FileReader();
            reader.onload = function (e) {
                const img = new Image();
                img.onload = function () {
                    const canvas = document.getElementById(im_bax);
                    const ctx = canvas.getContext("2d");
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                };
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);

            resolve(file);
        };

        // Trigger file picker
        fileInput.click();
    });
}


async function poseImages() {
    await uploadCanvasImage('1');
    await uploadCanvasImage('2');
}

async function uploadCanvasImage(a) {
    const statusEl = document.getElementById("poseStatus");
    statusEl.innerText = "Status: Preparing to upload...";
    a1 = a;
    if (a == '1') a1 = '1'
    else if (a == '2')  a1 = '2';
    else a1 = '3';

    if (a =='1') ref1 = 'ref_img1';
    else if (a == '2') ref1 = 'ref_img2';
    else ref1 = 'image1';
    
    value1 = calculateBinarySum();  
    if (value1 == 0) {
        statusEl.innerText = "Status: No angles selected";
        return;
    }

    const canvas = document.getElementById(ref1);
    if (!canvas) {
        statusEl.innerText = "Image not found";
        // console.error("Canvas not found");
        return;
    }
    statusEl.innerText = "Status: Preparing image..."+a1;
    // const canvas = document.getElementById("image1");
    const imageData = canvas.toDataURL('image/png');
    const ctx = canvas.getContext("2d");

    try {
        
        statusEl.innerText = "Status: 1";
        const response = await fetch("/upload-photo", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: imageData , "name":a1
                , "angle": value1 }) });
        
        statusEl.innerText = "Status: 2";
        const result = await response.json();
        const img = new Image();
        img.onload = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        };

        let text = "Status:\n";
        for (const [key, value] of Object.entries(result.angles)) {
            text += `${key}: ${value}\n`;
        }
        statusEl.innerText = text;
        
        img.src = result.processed_image;
        // statusEl.innerText = "Status: Image " + a1 + " processed successfully ✅";

    } catch (err) {
        console.error("Pose detection failed:", err);
        statusEl.innerText = "Sta+tus: Error"+err.toString();
    }
}
