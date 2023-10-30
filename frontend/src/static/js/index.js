let dropZone = document.getElementById('drop_zone');
let audioFileInput = document.getElementById('audio_file');
let fileList = document.getElementById('file_list');
let startRecordingButton = document.getElementById('start_recording');
let stopRecordingButton = document.getElementById('stop_recording');
let mediaRecorder;
let audioChunks = [];
let uploadedFiles = [];
let timerInterval;
let elapsedSeconds = 0;
let recordNumber = 0;

function appendFiles(newFiles) {
    for (let i = 0; i < newFiles.length; i++) {
        uploadedFiles.push(newFiles[i]);
    }
    displayFiles(uploadedFiles);
    updateUploadButtonState();
}

function updateUploadButtonState() {
    let uploadButton = document.getElementById('upload-button');
    uploadButton.disabled = uploadedFiles.length === 0;
}

dropZone.addEventListener('dragover', function(event) {
    event.preventDefault();
});

dropZone.addEventListener('drop', function(event) {
    event.preventDefault();
    let files = event.dataTransfer.files;
    appendFiles(files);
});

dropZone.addEventListener('click', function() {
    audioFileInput.click();
});

audioFileInput.addEventListener('change', function(event) {
    let files = event.target.files;
    appendFiles(files);
});

document.getElementById("upload-button").addEventListener('click', function(event) {
    var dataTransfer = new DataTransfer();
    uploadedFiles.forEach(file => dataTransfer.items.add(file));
    audioFileInput.files = dataTransfer.files;
});

startRecordingButton.addEventListener('click', function() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        clearInterval(timerInterval);
        startRecordingButton.innerHTML = '<i class="fas fa-microphone"></i> Start Recording';
    } else {
        navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];
            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = () => {
                recordNumber++;
                let audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                let file = new File([audioBlob], `RecordedAudio_${recordNumber}.wav`, { type: 'audio/wav' });
                appendFiles([file]);  // Append the new file to the existing array
            };

            // Start the timer
            elapsedSeconds = 0;
            startRecordingButton.innerHTML = '<i class="fas fa-microphone-slash"></i> Stop Recording <span class="timer">0:00</span>';
            let timerSpan = startRecordingButton.querySelector('.timer');
            timerInterval = setInterval(() => {
                elapsedSeconds++;
                let minutes = Math.floor(elapsedSeconds / 60);
                let seconds = elapsedSeconds % 60;
                timerSpan.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
            }, 1000);

            mediaRecorder.start();
        });
    }
});

function removeFile(index) {
    uploadedFiles.splice(index, 1);
    displayFiles(uploadedFiles);
    updateUploadButtonState();
}

function renameFile(index) {
    let newName = prompt("Enter new name:", uploadedFiles[index].name);
    if (newName) {
        let oldFile = uploadedFiles[index];
        let newFile = new File([oldFile], newName, { type: oldFile.type });
        uploadedFiles[index] = newFile;
        displayFiles(uploadedFiles);
    }
}


function displayFiles(files) {
    uploadedFiles = files;
    fileList.innerHTML = '';

    if (files.length === 0) {
        return;
    }

    let tableHTML = `
        <table id="file_table">
            <tbody>
    `;
    for (let i = 0; i < files.length; i++) {
        tableHTML += `
            <tr>
                <td class="filename-cell">${files[i].name}</td>
                <td class="button-cell">
                    <button class="file-button" onclick="removeFile(${i})"><i class="fas fa-trash-alt"></i> Remove</button>
                    <button class="file-button" onclick="renameFile(${i})"><i class="fas fa-pen"></i> Rename</button>
                </td>
            </tr>
        `;
    }
    tableHTML += `
            </tbody>
        </table>
    `;
    fileList.innerHTML = tableHTML;
}