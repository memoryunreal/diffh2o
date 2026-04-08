/* ========================================================================
   CRR-Flow Project Page - JavaScript
   ======================================================================== */

(function () {
  'use strict';

  /* ------------------------------------------------------------------
     1. Tab Switching
     ------------------------------------------------------------------ */
  function initTabs() {
    var tabGroups = document.querySelectorAll('.tab-group');
    tabGroups.forEach(function (group) {
      var targetSection = group.getAttribute('data-target');
      var buttons = group.querySelectorAll('.tab-btn');
      buttons.forEach(function (btn) {
        btn.addEventListener('click', function () {
          var tabId = btn.getAttribute('data-tab');

          // Update active button
          buttons.forEach(function (b) { b.classList.remove('active'); });
          btn.classList.add('active');

          // Show/hide containers
          var containers = document.querySelectorAll(
            '.' + targetSection + '-container'
          );
          containers.forEach(function (c) {
            if (c.getAttribute('data-id') === tabId) {
              c.classList.add('active');
              // Trigger lazy load for newly visible videos
              lazyLoadVideos(c);
              // Sync play state
              syncVideosInContainer(c);
            } else {
              c.classList.remove('active');
              // Pause hidden videos
              pauseVideosInContainer(c);
            }
          });
        });
      });
    });
  }

  /* ------------------------------------------------------------------
     2. Video Synchronization
     ------------------------------------------------------------------ */
  function syncVideosInContainer(container) {
    var videos = container.querySelectorAll('video');
    if (videos.length === 0) return;

    videos.forEach(function (video) {
      video.currentTime = 0;
      if (video.readyState >= 2) {
        video.play().catch(function () {});
      }
    });
  }

  function pauseVideosInContainer(container) {
    var videos = container.querySelectorAll('video');
    videos.forEach(function (v) { v.pause(); });
  }

  // Synchronize all videos within a comparison grid row
  function initVideoSync() {
    var grids = document.querySelectorAll('.video-compare-grid');
    grids.forEach(function (grid) {
      var videos = grid.querySelectorAll('video');
      if (videos.length < 2) return;

      videos.forEach(function (video) {
        video.addEventListener('play', function () {
          var currentTime = video.currentTime;
          videos.forEach(function (v) {
            if (v !== video) {
              v.currentTime = currentTime;
              v.play().catch(function () {});
            }
          });
        });

        video.addEventListener('pause', function () {
          videos.forEach(function (v) {
            if (v !== video) {
              v.pause();
            }
          });
        });

        video.addEventListener('seeked', function () {
          var currentTime = video.currentTime;
          videos.forEach(function (v) {
            if (v !== video) {
              v.currentTime = currentTime;
            }
          });
        });
      });
    });
  }

  /* ------------------------------------------------------------------
     3. Lazy Loading with IntersectionObserver
     ------------------------------------------------------------------ */
  function initLazyLoading() {
    if (!('IntersectionObserver' in window)) {
      // Fallback: load all videos immediately
      document.querySelectorAll('video[data-src]').forEach(function (v) {
        loadVideo(v);
      });
      return;
    }

    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          var container = entry.target;
          lazyLoadVideos(container);
          observer.unobserve(container);
        }
      });
    }, { rootMargin: '200px 0px' });

    // Observe each video comparison container
    document.querySelectorAll('.video-compare-container').forEach(function (c) {
      observer.observe(c);
    });

    // Also observe the long-horizon section
    var longHorizon = document.querySelector('.long-horizon-videos');
    if (longHorizon) {
      observer.observe(longHorizon);
    }
  }

  function lazyLoadVideos(container) {
    var videos = container.querySelectorAll('video[data-src]');
    videos.forEach(function (v) {
      loadVideo(v);
    });
  }

  function loadVideo(video) {
    var src = video.getAttribute('data-src');
    if (src && !video.getAttribute('src')) {
      video.setAttribute('src', src);
      video.removeAttribute('data-src');
      video.load();
    }
  }

  /* ------------------------------------------------------------------
     4. Copy BibTeX
     ------------------------------------------------------------------ */
  function initCopyBibtex() {
    var btn = document.getElementById('copy-bibtex');
    if (!btn) return;

    btn.addEventListener('click', function () {
      var pre = document.getElementById('bibtex-text');
      if (!pre) return;

      var text = pre.textContent;

      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function () {
          showCopied(btn);
        }).catch(function () {
          fallbackCopy(text, btn);
        });
      } else {
        fallbackCopy(text, btn);
      }
    });
  }

  function fallbackCopy(text, btn) {
    var textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.left = '-9999px';
    document.body.appendChild(textarea);
    textarea.select();
    try {
      document.execCommand('copy');
      showCopied(btn);
    } catch (e) {
      // silently fail
    }
    document.body.removeChild(textarea);
  }

  function showCopied(btn) {
    var original = btn.textContent;
    btn.textContent = 'Copied!';
    btn.classList.add('copied');
    setTimeout(function () {
      btn.textContent = original;
      btn.classList.remove('copied');
    }, 2000);
  }

  /* ------------------------------------------------------------------
     5. Auto-play visible active videos on load
     ------------------------------------------------------------------ */
  function autoplayActiveVideos() {
    document.querySelectorAll('.video-compare-container.active video').forEach(
      function (v) {
        if (v.getAttribute('src')) {
          v.play().catch(function () {});
        }
      }
    );
    // Long-horizon videos
    document.querySelectorAll('.long-horizon-videos video').forEach(
      function (v) {
        if (v.getAttribute('src')) {
          v.play().catch(function () {});
        }
      }
    );
  }

  /* ------------------------------------------------------------------
     Init
     ------------------------------------------------------------------ */
  document.addEventListener('DOMContentLoaded', function () {
    initTabs();
    initLazyLoading();
    initCopyBibtex();

    // Delay video sync setup until videos may have loaded
    setTimeout(function () {
      initVideoSync();
      autoplayActiveVideos();
    }, 500);
  });

})();
