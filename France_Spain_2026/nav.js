/* ============================================
   France & Spain 2026 — Shared Navigation + Utils
   ============================================ */
(function() {
  'use strict';

  // --- Inject site-wide navigation bar ---
  var currentPage = window.location.pathname.split('/').pop() || 'index.html';
  var tabs = [
    { href: 'index.html',          label: 'Overview' },
    { href: 'paris.html',          label: 'Paris' },
    { href: 'saint-emilion.html',  label: 'Saint-\u00c9milion' },
    { href: 'lescun.html',         label: 'Lescun' },
    { href: 'san-sebastian.html',  label: 'San Sebasti\u00e1n' },
    { href: 'travel-days.html',    label: 'Travel Days' },
    { href: 'packing.html',        label: 'Packing & Planning' }
  ];

  var navHTML = '<nav class="site-nav"><div class="site-nav-inner">';
  navHTML += '<span class="nav-title">France &amp; Spain 2026</span>';
  tabs.forEach(function(tab) {
    var active = (currentPage === tab.href) ? ' active' : '';
    navHTML += '<a href="' + tab.href + '" class="' + active + '">' + tab.label + '</a>';
  });
  navHTML += '</div></nav>';

  // Insert at top of body
  document.body.insertAdjacentHTML('afterbegin', navHTML);

  // --- Collapsible time-blocks (shared across all pages) ---
  document.querySelectorAll('.timeline .time-block').forEach(function(block) {
    var timeLabel = block.querySelector('.time-label');
    if (!timeLabel) return;

    var sumHTML = '';
    var bodyHTML = '';
    var title = '';

    var imagesHTML = '';
    Array.from(block.childNodes).forEach(function(node) {
      if (node.nodeType === 3) { sumHTML += node.textContent; return; }
      if (!node.tagName) return;
      var tag = node.tagName.toLowerCase();
      if (tag === 'span') {
        sumHTML += node.outerHTML + ' ';
        return;
      }

      // If the node contains images, extract them so they can be shown
      // outside the collapsed body while preserving the remaining text.
      var imgs = node.querySelectorAll && node.querySelectorAll('img');
      if (imgs && imgs.length) {
        imgs.forEach(function(img) { imagesHTML += img.outerHTML; });
        // create a shallow clone and remove images before serializing
        try {
          var clone = node.cloneNode(true);
          clone.querySelectorAll('img').forEach(function(i) { i.remove(); });
          if (clone.textContent && !title && (tag === 'p' || tag === 'ul')) {
            if (block.dataset.title) {
              title = block.dataset.title;
            } else {
              var strong = clone.querySelector && clone.querySelector('strong');
              if (strong) {
                title = strong.textContent.replace(/\.\s*$/, '');
              } else {
                var txt = clone.textContent.trim();
                if (txt.length > 50) txt = txt.substring(0, 47) + '\u2026';
                title = txt;
              }
            }
          }
          if (clone.outerHTML) bodyHTML += clone.outerHTML;
        } catch (e) {
          // fallback: include original node if cloning fails
          bodyHTML += node.outerHTML;
        }
      } else {
        if (!title && (tag === 'p' || tag === 'ul')) {
          if (block.dataset.title) {
            title = block.dataset.title;
          } else {
            var strong = node.querySelector && node.querySelector('strong');
            if (strong) {
              title = strong.textContent.replace(/\.\s*$/, '');
            } else {
              var txt = node.textContent.trim();
              if (txt.length > 50) txt = txt.substring(0, 47) + '\u2026';
              title = txt;
            }
          }
        }
        bodyHTML += node.outerHTML;
      }
    });

    // --- Ensure images inside existing <details> remain visible ---
    // For any <details> element on the page, move images that are not
    // inside the <summary> out into a sibling container so the images
    // remain visible whether the details are open or closed.
    document.querySelectorAll('details').forEach(function(det) {
      try {
        // skip if we've already created an adjacent image container
        if (det.nextElementSibling && det.nextElementSibling.classList.contains('timeline-images')) return;
        var imgs = Array.from(det.querySelectorAll('img'));
        if (!imgs.length) return;
        var container = document.createElement('div');
        container.className = 'timeline-images';
        imgs.forEach(function(img) {
          // don't move images that are inside the summary itself
          if (img.closest('summary')) return;
          container.appendChild(img);
        });
        if (container.children.length) {
          det.parentNode.insertBefore(container, det.nextSibling);
        }
      } catch (e) {
        /* ignore and continue on pages with unusual markup */
      }
    });

    if (!bodyHTML.trim()) return;

    if (title) {
      sumHTML += ' <span style="font-weight:600;color:var(--warm);font-size:0.92em;">' + title + '</span>';
    }

    var details = document.createElement('details');
    details.className = block.className;
    details.innerHTML = '<summary>' + sumHTML.trim() + '</summary><div class="tb-body">' + bodyHTML + '</div>';

    // If we extracted images, render them as a sibling element outside the
    // <details> so they remain visible whether the details are open or closed.
    if (imagesHTML.trim()) {
      var wrapper = document.createElement('div');
      wrapper.className = 'time-block-with-media';
      var imgContainer = document.createElement('div');
      imgContainer.className = 'timeline-images';
      imgContainer.innerHTML = imagesHTML;
      wrapper.appendChild(details);
      wrapper.appendChild(imgContainer);
      block.parentNode.replaceChild(wrapper, block);
    } else {
      block.parentNode.replaceChild(details, block);
    }
  });

  // --- Make destination leg-cards collapsible and compact ---
  document.querySelectorAll('dl.leg-card').forEach(function(card) {
    try {
      var titleEl = card.querySelector('h3');
      var titleHTML = titleEl ? titleEl.outerHTML : '';

      // Build a short summary from Dates and Lodging fields (if present)
      var divs = Array.from(card.querySelectorAll('div'));
      var dateText = '';
      var lodgingText = '';
      divs.forEach(function(d) {
        var dt = d.querySelector('dt');
        var dd = d.querySelector('dd');
        if (!dt || !dd) return;
        var key = dt.textContent.trim().toLowerCase();
        var val = dd.textContent.trim();
        if (!dateText && key.indexOf('date') === 0) dateText = val;
        if (!lodgingText && key.indexOf('lodg') === 0) lodgingText = val;
      });

      // Fix missing separators caused by <br> -> textContent concatenation
      if (lodgingText && /Hosted by/i.test(lodgingText)) {
        if (!/·\s*Hosted by/i.test(lodgingText)) {
          lodgingText = lodgingText.replace(/\s*Hosted by/i, ' · Hosted by');
        }
      }

      var summaryParts = [];
      if (titleEl) summaryParts.push(titleEl.textContent.trim());
      if (dateText) summaryParts.push(dateText);
      if (lodgingText) summaryParts.push(lodgingText.replace(/\s+/g,' '));

      // If the card contains images, use the first as a small thumbnail
      var img = card.querySelector('img');
      var imgHTML = '';
      if (img && img.src) {
        imgHTML = '<img class="dest-thumb" src="' + img.getAttribute('src') + '" alt="' + (titleEl ? titleEl.textContent.trim() : '') + ' image'> ';
      }

      var summaryHTML = imgHTML + '<strong style="margin-right:8px;">' + (summaryParts.shift() || '') + '</strong>' + (summaryParts.join(' · ') || '');

      // Create details wrapper
      var details = document.createElement('details');
      details.className = 'leg-card';
      details.innerHTML = '<summary>' + summaryHTML + '</summary><div class="leg-body">' + card.innerHTML.replace(/<h3[^>]*>.*?<\/h3>/i, '') + '</div>';

      // Replace original dl with details
      card.parentNode.replaceChild(details, card);
    } catch (e) {
      // if anything fails, leave the original card intact
      console.warn('Could not make leg-card collapsible', e);
    }
  });
})();
