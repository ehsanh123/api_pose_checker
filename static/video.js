let videoStream;
let pose;
let camera;

async function startCamera() {
            const video = document.getElementById('video');

            try {
                videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = videoStream;

                setupPose();
            } catch (err) {
                console.error("Error accessing camera:", err);
                alert("Could not access the camera.");
    }
}

function stopCamera() {
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }
    if (camera) {
        camera.stop();
    }
}
function takePhoto() {
            const video = document.getElementById('video');
            const canvas = document.getElementById('image1');
            const ctx = canvas.getContext('2d');

            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            const imageData = canvas.toDataURL('image/png');
            uploadToBackend(imageData);
        }