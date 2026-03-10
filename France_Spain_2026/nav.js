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

      // Build a short summary containing only: city name + (Hotel|Airbnb) + accommodation name
      var divs = Array.from(card.querySelectorAll('div'));
      var dateText = '';
      var lodgingText = '';
      var lodgingName = '';
      var lodgingType = '';
      divs.forEach(function(d) {
        var dt = d.querySelector('dt');
        var dd = d.querySelector('dd');
        if (!dt || !dd) return;
        var key = dt.textContent.trim().toLowerCase();
        var val = dd.textContent.trim();
        if (!dateText && key.indexOf('date') === 0) dateText = val;
        if (!lodgingText && key.indexOf('lodg') === 0) {
          lodgingText = val;
          // Prefer linked name (strong inside <a> or the <a> text)
          var a = d.querySelector('dd a');
          if (a) {
            var s = a.querySelector && a.querySelector('strong');
            lodgingName = (s ? s.textContent : a.textContent).trim();
            if (/airbnb\.com/i.test(a.getAttribute('href') || '')) lodgingType = 'Airbnb';
            else if (/hotel|h\u00f4tel/i.test(a.textContent || '')) lodgingType = 'Hotel';
            else lodgingType = '';
          } else {
            // Fallback: take the first text segment before a separator
            var txtNode = dd.childNodes && dd.childNodes.length ? (dd.childNodes[0].textContent || dd.textContent) : dd.textContent;
            if (txtNode) {
              var tname = txtNode.split('·')[0].split('\n')[0].trim();
              lodgingName = tname;
            }
            if (/hotel|h\u00f4tel/i.test(val)) lodgingType = 'Hotel';
            else if (/airbnb/i.test(val)) lodgingType = 'Airbnb';
          }
        }
      });

      // Keep full lodging text intact in the expanded body, but normalize separators there
      if (lodgingText && /Hosted by/i.test(lodgingText)) {
        if (!/·\s*Hosted by/i.test(lodgingText)) {
          lodgingText = lodgingText.replace(/\s*Hosted by/i, ' · Hosted by');
        }
      }

      var lines = [];
      if (titleEl) lines.push(titleEl.textContent.trim());
      if (dateText) {
        var dparts = dateText.split('·').map(function(s){ return s.trim(); });
        dparts.forEach(function(p){ if (p) lines.push(p); });
      }
      var lodgingSummary = '';
      if (lodgingName) {
        lodgingSummary = (lodgingType ? (lodgingType + ': ') : '') + lodgingName;
      }
      if (lodgingSummary) lines.push(lodgingSummary);

      // If there's an image in the card, extract it and use as a small thumbnail
      var thumbHTML = '';
      try {
        var imgEl = card.querySelector('img');
        if (imgEl && imgEl.getAttribute('src')) {
          var src = imgEl.getAttribute('src');
          var alt = imgEl.getAttribute('alt') || '';
          // wrap thumbnail in a link so it can be clicked to open the full image
          thumbHTML = '<a href="' + src + '" class="dest-thumb-link"><img class="dest-thumb" src="' + src + '" alt="' + alt + '"></a>';
          // remove original image from the card so it doesn't duplicate in body
          imgEl.parentNode && imgEl.parentNode.removeChild(imgEl);
        }
      } catch (e) {
        // ignore image extraction errors
      }

      var firstLine = (lines.shift() || '');
      var restHtml = lines.map(function(l){ return '<span class="summary-line" style="display:block;color:var(--muted);font-size:0.95em;">' + l + '</span>'; }).join('');
      var summaryMain = '<div class="summary-block"><strong style="display:block;margin-bottom:4px;">' + firstLine + '</strong>' + restHtml + '</div>';
      var summaryHTML = (thumbHTML ? thumbHTML : '') + '<span>' + summaryMain + '</span>';

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

  // --- Build a simple month calendar showing trip dates ---
  (function(){
    // helper: parse date range like "May 13–16 · 3 nights"
    function parseDateRange(txt){
      if (!txt) return null;
      var part = txt.split('\u00b7')[0].trim(); // before middot
      // try pattern: Month D–D or Month D–Month D
      var m = part.match(/([A-Za-z]+)\s*(\d{1,2})\s*[–-]\s*([A-Za-z]*\s*\d{1,2})/);
      if (!m) {
        // maybe single day: "May 25–26" fallback
        var single = part.match(/([A-Za-z]+)\s*(\d{1,2})/);
        if (!single) return null;
        var mon = single[1], day = parseInt(single[2],10);
        var d = new Date(2026, new Date(Date.parse(mon + ' 1, 2026')).getMonth(), day);
        return [d,d];
      }
      var startMon = m[1].trim();
      var startDay = parseInt(m[2],10);
      var endRaw = m[3].trim();
      var endMon = startMon;
      var endDay = null;
      var em = endRaw.match(/([A-Za-z]+)\s*(\d{1,2})/);
      if (em) { endMon = em[1].trim(); endDay = parseInt(em[2],10); }
      else { endDay = parseInt(endRaw,10); }
      var s = new Date(2026, new Date(Date.parse(startMon + ' 1, 2026')).getMonth(), startDay);
      var e = new Date(2026, new Date(Date.parse(endMon + ' 1, 2026')).getMonth(), endDay);
      return [s,e];
    }

    // collect events from the rendered leg-cards
    var events = [];
    function cleanTitle(t){ if(!t) return t; return t.split(/\s*[–—-]\s*/)[0].trim(); }
    document.querySelectorAll('details.leg-card').forEach(function(det){
      try {
        var titleRaw = det.querySelector('summary strong') ? det.querySelector('summary strong').textContent.trim() : (det.querySelector('summary') && det.querySelector('summary').textContent.trim());
        var title = cleanTitle(titleRaw) || 'Trip';
        var dateDt = Array.from(det.querySelectorAll('.leg-body dt')).find(function(d){ return /date/i.test(d.textContent); });
        var dateTxt = dateDt ? (dateDt.nextElementSibling && dateDt.nextElementSibling.textContent.trim()) : null;
        var range = parseDateRange(dateTxt);
        if (range) events.push({ title: title, start: range[0], end: range[1] });
      } catch(e){ /* ignore */ }
    });

    // manual override events (specific flights / returns)
    function makeDate(y,m,d){ return new Date(y,m,d); }
    // May 12 DEN-CDG Flight
    events.push({ title: 'DEN-CDG Flight', start: makeDate(2026,4,12), end: makeDate(2026,4,12) });
    // May 25 Return to Paris
    events.push({ title: 'Return to Paris', start: makeDate(2026,4,25), end: makeDate(2026,4,25) });
    // May 26 CDG-DEN Flight
    events.push({ title: 'CDG-DEN Flight', start: makeDate(2026,4,26), end: makeDate(2026,4,26) });

    if (!events.length) return;

    // color palette and mapping
    var palette = ['#4a1a2e','#4a6e8b','#8b6f4e','#e07a5f','#2a9d8f','#f4a261','#6a6f8c'];
    var map = {};
    var pi = 0;

    // assign colors (special-case Return -> flightColor)
    events.forEach(function(ev){
      var key = ev.title;
      if (/Return|Return to|Departure|flight|Flight|Return/i.test(key)) { map[key] = '#999'; return; }
      if (!map[key]) { map[key] = palette[pi++ % palette.length]; }
    });

    // render calendar for May 2026
    var year = 2026, month = 4; // May (0-based)
    var first = new Date(year, month, 1);
    var last = new Date(year, month+1, 0);
    var container = document.getElementById('trip-calendar');
    if (!container) return;
    container.innerHTML = '';

    // day-of-week headers (Sun..Sat)
    var days = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
    days.forEach(function(d){ var h = document.createElement('div'); h.className='day-header'; h.textContent = d; container.appendChild(h); });

    // determine leading blanks
    var lead = first.getDay();
    for (var i=0;i<lead;i++){ var blank = document.createElement('div'); blank.className='cal-day'; blank.innerHTML=''; container.appendChild(blank); }

    for (var d=1; d<=last.getDate(); d++){
      var cell = document.createElement('div'); cell.className='cal-day';
      cell.dataset.day = d;
      var dn = document.createElement('div'); dn.className='day-num'; dn.textContent = d; cell.appendChild(dn);

      // choose a single event to represent the day (first matching event)
      var cur = new Date(year, month, d);
      var matched = null;
      for (var ei=0; ei<events.length; ei++){
        var ev = events[ei];
        var sd = new Date(ev.start.getFullYear(), ev.start.getMonth(), ev.start.getDate());
        var ed = new Date(ev.end.getFullYear(), ev.end.getMonth(), ev.end.getDate());
        if (cur >= sd && cur <= ed) { matched = ev; break; }
      }
      if (matched) {
        var color = map[matched.title] || '#888';
        // color entire day square
        cell.style.background = color;
        cell.style.color = '#fff';
        cell.style.borderColor = "rgba(0,0,0,0.06)";
        var label = document.createElement('div');
        label.className = 'event-label';
        label.style.color = '#fff';
        label.style.fontWeight = '700';
        label.style.marginTop = '18px';
        label.style.fontSize = '0.82em';
        label.style.overflow = 'hidden';
        label.style.textOverflow = 'ellipsis';
        label.style.whiteSpace = 'nowrap';
        // shorten title if necessary
        var t = matched.title || '';
        if (t.length > 18) t = t.substring(0,15) + '\u2026';
        label.textContent = t;
        cell.appendChild(label);
      }

      container.appendChild(cell);
    }

    // legend
    var legend = document.getElementById('calendar-legend');
    if (legend){
      legend.innerHTML = '';
      Object.keys(map).forEach(function(k){
        var item = document.createElement('div'); item.className='legend-item';
        var sw = document.createElement('span'); sw.className='legend-swatch'; sw.style.background = map[k];
        var lbl = document.createElement('span'); lbl.textContent = k;
        item.appendChild(sw); item.appendChild(lbl);
        legend.appendChild(item);
      });
      // small note about colors
      var note = document.createElement('div'); note.style.marginLeft='8px'; note.style.color='var(--muted)'; note.style.fontSize='0.9em'; note.textContent='(colors: stays and travel)'; legend.appendChild(note);
    }
  })();

  // --- Lightbox behavior for destination thumbnails ---
  // Create a single reusable lightbox element and attach click handlers
  (function(){
    var lb = document.createElement('div');
    lb.className = 'lightbox';
    lb.innerHTML = '<span class="close">✕</span><img src="" alt="">';
    document.body.appendChild(lb);
    var lbImg = lb.querySelector('img');
    var lbClose = lb.querySelector('.close');

    function show(src, alt){
      lbImg.src = src;
      lbImg.alt = alt || '';
      lb.style.display = 'flex';
      document.body.style.overflow = 'hidden';
    }
    function hide(){
      lb.style.display = 'none';
      lbImg.src = '';
      document.body.style.overflow = '';
    }

    document.addEventListener('click', function(e){
      var a = e.target.closest && e.target.closest('.dest-thumb-link');
      if (a) {
        e.preventDefault();
        show(a.href, a.querySelector('img') && a.querySelector('img').alt);
        return;
      }
      if (e.target === lb || e.target === lbClose) hide();
    });
    document.addEventListener('keydown', function(e){ if (e.key === 'Escape') hide(); });
  })();
})();
