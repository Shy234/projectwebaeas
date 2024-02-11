function toggleFileForm(folderId) {

    var form = document.getElementById('form' + folderId);
    if (form.style.display === 'none') {
        form.style.display = 'block';
    } else {
        form.style.display = 'none';
    }
}
function deleteFolder(folderId) {
    fetch("/delete-folder", {
      method: "POST",
      body: JSON.stringify({ folderId: folderId }),
      headers: {
        'Content-Type': 'application/json'
      }
    }).then((_res) => {
      window.location.href = "/";
    });
  }
  
  function deleteFile(folderId, fileId) {
    if (confirm("Are you sure you want to delete this file?")) {
        fetch(`/delete-file/${folderId}/${fileId}`, {
            method: "POST",
            headers: {
                'Content-Type': 'application/json'
            }
        }).then((_res) => {
            window.location.reload();
        });
    }
}


//essay view
function loadFile() {
  const fileInput = document.getElementById('fileInput');
  const file = fileInput.files[0];

  if (file) {
      const reader = new FileReader();
      
      reader.onload = function (e) {
          const textarea = document.getElementById('essay');
          const fileType = file.name.split('.').pop().toLowerCase();

          if (fileType === 'pdf') {
              
              readPdf(file, textarea);
          } else if (fileType === 'doc' || fileType === 'docx') {
              // Handle DOC files (text extraction)
              readDoc(file, textarea);
          } else {
              textarea.value = e.target.result;
          }
      };

      if (file.type === 'application/pdf') {
          reader.readAsArrayBuffer(file);
      } else {
          reader.readAsText(file);
      }
  }
}

function readPdf(file, textarea) {
  const reader = new FileReader();

  reader.onload = function (e) {
      const arrayBuffer = e.target.result;
      pdfjsLib.getDocument(arrayBuffer).then(function(pdf) {
          let text = '';
          const numPages = pdf.numPages;

          for (let pageNum = 1; pageNum <= numPages; pageNum++) {
              pdf.getPage(pageNum).then(function(page) {
                  return page.getTextContent();
              }).then(function(content) {
                  content.items.forEach(function(item) {
                      text += item.str + ' ';
                  });
                  text += '\n';
                  textarea.value = text;
              });
          }
      });
  };

  reader.readAsArrayBuffer(file);
}

function readDoc(file, textarea) {
  Mammoth.extractRawText({ arrayBuffer: file }, {})
      .then(function (result) {
          textarea.value = result.value;
      })
      .catch(function (err) {
          console.error(err);
      });
}

function viewFileContent() {
  const fileInput = document.getElementById('fileInput');
  const file = fileInput.files[0];

  if (file) {
      loadFile(); 
  } else {
      alert("Please select a file first.");
  }
}
// selection
const selectBtn = document.querySelector(".select-btn"),
      items = document.querySelectorAll(".item");

selectBtn.addEventListener("click", () => {
    selectBtn.classList.toggle("open");
});

items.forEach(item => {
    item.addEventListener("click", () => {
        item.classList.toggle("checked");

        let checked = document.querySelectorAll(".checked"),
            btnText = document.querySelector(".btn-text");

            if(checked && checked.length > 0){
                btnText.innerText = `${checked.length} Selected`;
            }else{
                btnText.innerText = "Select criteria";
            }
    });
})

function selectFolder(element) {
    
    var folderId = element.getAttribute('data-folder-id');

    saveData(folderId);
    element.classList.add('selected');
  }

  function saveData(folderId) {
    console.log('Saving data for folder with id:', folderId);
  }



