document.addEventListener('DOMContentLoaded', () => {
    // 1. Search Tabs Logic
    const searchTabs = document.querySelectorAll('.search-tabs .tab-btn');
    const textContainer = document.getElementById('text-search-container');
    const imgContainer = document.getElementById('image-search-container');

    searchTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            searchTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            const containers = document.querySelectorAll('.search-input-container');
            containers.forEach(c => c.style.display = 'none');
            
            const targetId = `${tab.dataset.searchType}-search-container`;
            const target = document.getElementById(targetId);
            if(target) target.style.display = 'flex';
        });
    });

    // 2. Image Dropzone Logic
    const dropzone = document.getElementById('image-dropzone');
    const imageInput = document.getElementById('image-input');
    const imagePreview = document.getElementById('image-preview');
    const searchImgBtn = document.getElementById('search-image-btn');
    let currentImageFile = null;

    dropzone.addEventListener('click', () => imageInput.click());
    
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });
    
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleImageSelection(e.dataTransfer.files[0]);
        }
    });

    imageInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleImageSelection(e.target.files[0]);
        }
    });

    function handleImageSelection(file) {
        if (!file.type.startsWith('image/')) return;
        currentImageFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.classList.remove('hidden');
            searchImgBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // 2.1 Video/Audio Handlers
    ['video', 'audio'].forEach(type => {
        const dropzone = document.getElementById(`${type}-dropzone`);
        const input = document.getElementById(`${type}-input`);
        const placeholder = document.getElementById(`${type}-preview-placeholder`);
        
        function handleFileSelection(files) {
            if (!files.length) return;
            const file = files[0];
            input.files = files; // Sync the files
            
            document.getElementById(`search-${type}-btn`).disabled = false;
            
            if (type === 'video') {
                const videoPreview = document.getElementById('video-preview');
                if (videoPreview) {
                    videoPreview.src = URL.createObjectURL(file);
                    videoPreview.classList.remove('hidden');
                    placeholder.classList.add('hidden');
                }
            } else {
                placeholder.textContent = `Audio selected: ${file.name}`;
                placeholder.classList.remove('hidden');
            }
        }

        dropzone.addEventListener('click', () => input.click());
        dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
        dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            handleFileSelection(e.dataTransfer.files);
        });
        input.addEventListener('change', (e) => {
            handleFileSelection(e.target.files);
        });
    });

    // 3. Results Navigation — scroll to section on click
    const navBtns = document.querySelectorAll('.r-tab-btn');

    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetId = btn.dataset.scrollTo;
            const targetEl = document.getElementById(targetId);
            if (targetEl) {
                // Offset for the sticky nav bar height
                const navHeight = document.querySelector('.results-nav').offsetHeight;
                const targetPos = targetEl.closest('.modality-section').offsetTop - navHeight - 20;
                window.scrollTo({ top: targetPos, behavior: 'smooth' });
            }
        });
    });

    // 4. API Calls & Rendering
    const searchTextBtn = document.getElementById('search-text-btn');
    const textQuery = document.getElementById('text-query');
    const loading = document.getElementById('loading');
    const resultsSection = document.getElementById('results-section');
    
    const imgResults = document.getElementById('image-results');
    const vidResults = document.getElementById('video-results');
    const audResults = document.getElementById('audio-results');
    const imgCount = document.getElementById('img-count');
    const vidCount = document.getElementById('vid-count');
    const audCount = document.getElementById('aud-count');

    // Trigger search on enter key
    textQuery.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') searchTextBtn.click();
    });

    searchTextBtn.addEventListener('click', async () => {
        const query = textQuery.value.trim();
        if(!query) return;
        
        await performSearch('/api/search/text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });
    });

    searchImgBtn.addEventListener('click', async () => {
        if(!currentImageFile) return;
        
        const formData = new FormData();
        formData.append('file', currentImageFile);
        
        await performSearch('/api/search/image', {
            method: 'POST',
            body: formData
        });
    });

    document.getElementById('search-video-btn').addEventListener('click', async () => {
        const fileInput = document.getElementById('video-input');
        if (!fileInput.files.length) return;
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        await performSearch('/api/search/video', {
            method: 'POST',
            body: formData
        });
    });

    document.getElementById('search-audio-btn').addEventListener('click', async () => {
        const fileInput = document.getElementById('audio-input');
        if (!fileInput.files.length) return;
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        await performSearch('/api/search/audio', {
            method: 'POST',
            body: formData
        });
    });

    async function performSearch(endpoint, options) {
        loading.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        
        try {
            const res = await fetch(endpoint, options);
            const data = await res.json();
            renderResults(data);
            resultsSection.classList.remove('hidden');
            
            // Toggle centered to top-aligned state
            document.querySelector('.container').classList.add('results-showing');
            
            // Wait for transition to start, then scroll precisely to images
            setTimeout(() => {
                const targetEl = document.getElementById('image-results-section');
                if (targetEl) {
                    const navHeight = document.querySelector('.results-nav').offsetHeight;
                    const targetPos = targetEl.offsetTop - navHeight - 20;
                    window.scrollTo({ top: targetPos, behavior: 'smooth' });
                }
            }, 100);
            
        } catch (e) {
            console.error("Search failed:", e);
            alert("Search failed. Check console for details.");
        } finally {
            loading.classList.add('hidden');
        }
    }

    function renderResults(data) {
        // Clear old
        imgResults.innerHTML = '';
        vidResults.innerHTML = '';
        audResults.innerHTML = '';
        
        // Update badges
        imgCount.textContent = data.image ? data.image.length : 0;
        vidCount.textContent = data.video ? data.video.length : 0;
        audCount.textContent = data.audio ? data.audio.length : 0;

        // Render Images
        if (data.image) {
            data.image.forEach((item, idx) => {
                const url = window.location.origin + item.url;
                imgResults.innerHTML += `
                    <div class="media-card">
                        <img src="${url}" alt="Result ${idx + 1}" loading="lazy">
                        <div class="card-info">
                            <span class="result-number">#${idx + 1}</span>
                            <span class="score">${item.score.toFixed(4)}</span>
                        </div>
                    </div>
                `;
            });
        }

        // Render Videos
        if (data.video) {
            data.video.forEach((item, idx) => {
                const url = window.location.origin + item.url;
                vidResults.innerHTML += `
                    <div class="list-item">
                        <div class="list-number">#${idx + 1}</div>
                        <div class="list-media">
                            <video src="${url}" controls preload="metadata"></video>
                        </div>
                        <div class="list-details">
                            <span class="score">Similarity: ${item.score.toFixed(4)}</span>
                        </div>
                    </div>
                `;
            });
        }

        // Render Audio
        if (data.audio) {
            data.audio.forEach((item, idx) => {
                const url = window.location.origin + item.url;
                audResults.innerHTML += `
                    <div class="list-item">
                        <div class="list-number">#${idx + 1}</div>
                        <div class="list-media">
                            <audio src="${url}" controls preload="metadata" style="width: 100%;"></audio>
                        </div>
                        <div class="list-details">
                            <span class="score">Similarity: ${item.score.toFixed(4)}</span>
                        </div>
                    </div>
                `;
            });
        }
    }
});
