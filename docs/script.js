/* ===================================
   Selective Steering - JavaScript
   =================================== */

// Copy BibTeX to clipboard
function copyBibtex() {
    const bibtexText = document.getElementById('bibtex').innerText;
    navigator.clipboard.writeText(bibtexText).then(() => {
        const btn = document.querySelector('.copy-btn span');
        const originalText = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => {
            btn.textContent = originalText;
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy: ', err);
    });
}

// Tab switching for results section
document.addEventListener('DOMContentLoaded', () => {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active class from all buttons and contents
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            // Add active class to clicked button and corresponding content
            btn.classList.add('active');
            const tabId = btn.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });

    // Smooth scroll for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;

            const target = document.querySelector(targetId);
            if (target) {
                const navHeight = document.querySelector('.navbar').offsetHeight;
                const targetPosition = target.getBoundingClientRect().top + window.pageYOffset - navHeight - 20;

                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });

    // Navbar scroll effect
    const navbar = document.getElementById('navbar');
    let lastScrollY = window.scrollY;

    window.addEventListener('scroll', () => {
        if (window.scrollY > 100) {
            navbar.style.background = 'rgba(15, 15, 35, 0.95)';
        } else {
            navbar.style.background = 'rgba(15, 15, 35, 0.9)';
        }
        lastScrollY = window.scrollY;
    });

    // Intersection Observer for fade-in animations
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };

    const fadeInElements = document.querySelectorAll('.contribution-card, .method-grid, .algorithm-box');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    fadeInElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
});

// Image lightbox (simple implementation)
document.querySelectorAll('.method-figure img').forEach(img => {
    img.style.cursor = 'zoom-in';
    img.addEventListener('click', () => {
        const overlay = document.createElement('div');
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.9);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 9999;
            cursor: zoom-out;
            padding: 2rem;
        `;

        const enlargedImg = document.createElement('img');
        enlargedImg.src = img.src;
        enlargedImg.style.cssText = `
            max-width: 90%;
            max-height: 90vh;
            border-radius: 8px;
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5);
        `;

        overlay.appendChild(enlargedImg);
        document.body.appendChild(overlay);

        overlay.addEventListener('click', () => {
            overlay.remove();
        });

        document.addEventListener('keydown', function escHandler(e) {
            if (e.key === 'Escape') {
                overlay.remove();
                document.removeEventListener('keydown', escHandler);
            }
        });
    });
});

// ===================================
// Examples Viewer
// ===================================

let currentExamples = [];
let currentExampleIndex = 0;

// Load examples for a given degree
async function loadExamples(degree) {
    const container = document.getElementById('examples-container');
    const meta = document.getElementById('examples-meta');

    container.innerHTML = '<p class="loading-text">Loading examples...</p>';

    try {
        const response = await fetch(`examples/results_degree_${degree}.json`);
        if (!response.ok) throw new Error('Failed to load');

        const data = await response.json();
        currentExamples = data.results;
        currentExampleIndex = 0;

        // Update metadata
        meta.innerHTML = `Avg. Perplexity: ${data.metadata.average_perplexity.toFixed(3)} | ${data.metadata.num_samples} samples`;

        // Display first example
        displayExample(currentExampleIndex);
        updateNavigation();
    } catch (error) {
        container.innerHTML = '<p class="loading-text">Failed to load examples. Please try again.</p>';
        console.error('Error loading examples:', error);
    }
}

// Display a single example
function displayExample(index) {
    const container = document.getElementById('examples-container');
    const example = currentExamples[index];

    if (!example) {
        container.innerHTML = '<p class="loading-text">No examples available.</p>';
        return;
    }

    // Truncate response if too long
    const maxLength = 1500;
    let responseText = example.response;
    if (responseText.length > maxLength) {
        responseText = responseText.substring(0, maxLength) + '...';
    }

    container.innerHTML = `
        <div class="example-item">
            <div class="example-prompt">
                <strong>Prompt</strong>
                <p>${escapeHtml(example.prompt)}</p>
            </div>
            <div class="example-response">
                <strong>Model Response</strong>
                <p>${escapeHtml(responseText)}</p>
                <div class="example-perplexity">Perplexity: ${example.perplexity.toFixed(4)}</div>
            </div>
        </div>
    `;
}

// Update navigation buttons and counter
function updateNavigation() {
    const prevBtn = document.getElementById('prev-example');
    const nextBtn = document.getElementById('next-example');
    const counter = document.getElementById('example-counter');

    prevBtn.disabled = currentExampleIndex === 0;
    nextBtn.disabled = currentExampleIndex >= currentExamples.length - 1;
    counter.textContent = `${currentExampleIndex + 1} / ${currentExamples.length}`;
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initialize examples viewer
document.addEventListener('DOMContentLoaded', () => {
    const degreeSelect = document.getElementById('degree-select');
    const prevBtn = document.getElementById('prev-example');
    const nextBtn = document.getElementById('next-example');

    if (degreeSelect) {
        // Load initial examples
        loadExamples(degreeSelect.value);

        // Handle degree change
        degreeSelect.addEventListener('change', (e) => {
            loadExamples(e.target.value);
        });

        // Handle navigation
        prevBtn.addEventListener('click', () => {
            if (currentExampleIndex > 0) {
                currentExampleIndex--;
                displayExample(currentExampleIndex);
                updateNavigation();
            }
        });

        nextBtn.addEventListener('click', () => {
            if (currentExampleIndex < currentExamples.length - 1) {
                currentExampleIndex++;
                displayExample(currentExampleIndex);
                updateNavigation();
            }
        });
    }
});
