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

  // --- global map resize helper ---
  // Leaflet sometimes renders grey tiles when the container is hidden or not yet laid out.
  // Run a pass after the window loads to force any existing maps to redraw.  Individual
  // pages may also call invalidateSize in more specific contexts (e.g. toggling details).
  // attempt to rebuild maps entirely if the grey patch persists
  function rebuildMapElement(el) {
    var mm = el._leaflet_map;
    if (!mm) return;
    var center = mm.getCenter();
    var zoom = mm.getZoom();
    // capture tile layers and other layers so we can re-add them
    var saved = [];
    mm.eachLayer(function(l) { saved.push(l); });
    // remove old map instance
    mm.remove();
    var newm = L.map(el.id, {scrollWheelZoom: true}).setView(center, zoom);
    // reattach tile layers first, then everything else
    saved.forEach(function(l) { newm.addLayer(l); });
    // swap reference so later code still works
    el._leaflet_map = newm;
  }

  window.addEventListener('load', function(){
    var doResize = function(){
      document.querySelectorAll('.leaflet-container').forEach(function(el){
        var mm = el._leaflet_map;
        if(!mm) return;
        // ensure container size is correct, force view reset too
        mm.invalidateSize(true);
        try {
          mm.setView(mm.getCenter(), mm.getZoom(), {animate:false});
        } catch(e) {}
        // force tile layers to refresh (fix grey/blank patches)
        mm.eachLayer(function(layer){
          if(layer instanceof L.TileLayer && layer.redraw) layer.redraw();
        });
        // tiny pan/push to coax any straggling tiles to load
        try {
          mm.panBy([2,0], {animate:false});
          mm.panBy([-2,0], {animate:false});
        } catch(e){/* ignore */}
      });
    };
    // run immediately, then again after layout settles
    doResize();
    setTimeout(doResize, 200);
    setTimeout(doResize, 1000);
    // final brutal pass: rebuild the map entirely in case anything
    // at the low level was corrupted.  this runs after the other delays
    setTimeout(function(){
      document.querySelectorAll('.leaflet-container').forEach(rebuildMapElement);
    }, 300);
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
    // Explicit stay ranges (override parsed values to ensure accuracy)
    events.push({ title: 'Paris', start: makeDate(2026,4,13), end: makeDate(2026,4,15) });
    events.push({ title: 'Saint-Émilion', start: makeDate(2026,4,16), end: makeDate(2026,4,18) });
    events.push({ title: 'Lescun', start: makeDate(2026,4,19), end: makeDate(2026,4,21) });
    events.push({ title: 'San Sebastian', start: makeDate(2026,4,22), end: makeDate(2026,4,24) });

    if (!events.length) return;

    // color palette and mapping
    // softer / paler color palette for calendar days
    var palette = ['#e8cfd2','#cfe8eb','#efe8cf','#f6dccb','#cfeee0','#efe9ff','#ecebf4'];
    var map = {};
    var pi = 0;

    // assign fixed colors for known stays so they look consistent
    map['Paris'] = palette[0];
    map['Saint-\u00c9milion'] = palette[1];
    map['Saint-Emilion'] = palette[1];
    map['Lescun'] = palette[2];
    map['San Sebastian'] = palette[3];
    map['San Sebasti\u00e1n'] = palette[3];

    // assign colors (special-case Return -> flightColor)
    events.forEach(function(ev){
      var key = ev.title;
      // make flight/return days a light gray
      if (/Return|Return to|Departure|flight|Flight|Return/i.test(key)) { map[key] = '#cfcfcf'; return; }
      if (!map[key]) { map[key] = palette[pi++ % palette.length]; }
    });

    // render calendar for May 2026
    var year = 2026, month = 4; // May (0-based)
    var first = new Date(year, month, 1);
    var last = new Date(year, month+1, 0);
    var container = document.getElementById('trip-calendar');
    if (!container) return;
    container.innerHTML = '';

    // find reusable SVG icons from the route timeline (plane, train, car)
    var svgIconSrc = {};
    try {
      svgIconSrc.plane = document.querySelector('.rt-leg.flight .rt-icon svg') || document.querySelector('.route-timeline .flight .rt-icon svg') || null;
      svgIconSrc.train = document.querySelector('.rt-leg.train .rt-icon svg') || document.querySelector('.route-timeline .train .rt-icon svg') || null;
      svgIconSrc.car   = document.querySelector('.rt-leg.car .rt-icon svg')   || document.querySelector('.route-timeline .car .rt-icon svg')   || null;
    } catch(e){ svgIconSrc = {plane:null,train:null,car:null}; }

    // explicit per-day color overrides and icon assignments
    var dateColorOverrides = {
      // force May 16 to Saint-Émilion color, May 22 to San Sebastian color
      16: map['Saint-Émilion'] || map['Saint-Emilion'],
      22: map['San Sebastian'] || map['San Sebasti\u00e1n']
    };

    // use logical keys that map to the SVG sources above; fall back to emoji if SVG missing
    var dateIconOverrides = {
      12: ['plane'],
      16: ['train','car'],
      19: ['car'],
      22: ['car'],
      25: ['car','train'],
      26: ['plane']
    };

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
      // collect all events that match this day, then pick by priority
      var matches = [];
      for (var ei=0; ei<events.length; ei++){
        var ev = events[ei];
        var sd = new Date(ev.start.getFullYear(), ev.start.getMonth(), ev.start.getDate());
        var ed = new Date(ev.end.getFullYear(), ev.end.getMonth(), ev.end.getDate());
        if (cur >= sd && cur <= ed) { matches.push(ev); }
      }
      if (matches.length) {
        // priority: stays (named destinations) first, then returns/flights, then first match
        var preferNames = ['Paris','Saint-Émilion','Saint-Emilion','Lescun','San Sebastian','San Sebastián'];
        function pickBest(list){
          for (var i=0;i<preferNames.length;i++){
            for (var j=0;j<list.length;j++) if ((list[j].title||'').indexOf(preferNames[i]) !== -1) return list[j];
          }
          for (var j=0;j<list.length;j++) if (/Return|Return to|Departure|flight|Flight|Return/i.test(list[j].title)) return list[j];
          return list[0];
        }
        var matched = pickBest(matches);
        var color = map[matched.title] || '#888';
        // color entire day square
        cell.style.background = color;
        cell.style.borderColor = "rgba(0,0,0,0.06)";
        var label = document.createElement('div');
        label.className = 'event-label';
        // label text uses dark color to read on pale backgrounds
        label.style.color = '#333';
        label.style.fontWeight = '400';
        label.style.marginTop = '18px';
        label.style.fontSize = '0.82em';
        label.style.overflow = 'hidden';
        label.style.textOverflow = 'ellipsis';
        label.style.whiteSpace = 'nowrap';
        // date-based label overrides (avoid ellipses for these key dates)
        var dKey = cur.getDate();
        var overrideMap = {
          25: 'Return to Paris',
          26: 'CDG-DEN Flight',
          16: 'Saint-Émilion',
          22: 'San Sebastian'
        };
        var t = matched.title || '';
        if (overrideMap[dKey]) {
          t = overrideMap[dKey];
        } else {
          // keep reasonable length but avoid ellipses where possible
          if (t.length > 22) t = t.substring(0,22);
        }

        // apply any per-day color overrides (fixes for May 16 and May 22)
        if (dateColorOverrides && dateColorOverrides[dKey]) {
          color = dateColorOverrides[dKey];
          cell.style.background = color;
        }

        // render icons for specific days — clone SVGs from the timeline when available
        if (dateIconOverrides && dateIconOverrides[dKey]) {
          var icons = dateIconOverrides[dKey];
          var iconsEl = document.createElement('div');
          iconsEl.className = 'day-icons';
          iconsEl.setAttribute('aria-hidden','true');
          iconsEl.style.display = 'inline-flex';
          iconsEl.style.gap = '6px';
          iconsEl.style.alignItems = 'center';
          iconsEl.style.marginTop = '6px';

          icons.forEach(function(ic){
            var appended = false;
            if (ic === 'plane' || ic === 'train' || ic === 'car') {
              var src = svgIconSrc[ic];
              if (src) {
                try {
                  var clone = src.cloneNode(true);
                  // normalize size for calendar: small square icon
                  clone.removeAttribute('class');
                  clone.style.width = '24px';
                  clone.style.height = '24px';
                  clone.style.display = 'inline-block';
                  clone.style.verticalAlign = 'middle';
                  if (ic === 'plane') {
                    try { clone.style.fill = '#666666'; } catch(e) {}
                    try { clone.style.stroke = '#666666'; } catch(e) {}
                  }
                  iconsEl.appendChild(clone);
                  appended = true;
                } catch(e) { appended = false; }
              }
            }
            if (!appended) {
              // fallback to emoji if SVG not present
              var span = document.createElement('span');
              span.style.fontSize = '24px';
              span.textContent = (ic === 'plane' ? '✈️' : (ic === 'train' ? '🚆' : (ic === 'car' ? '🚕' : ic)) );
              iconsEl.appendChild(span);
            }
          });

          cell.appendChild(iconsEl);
          // nudge the label down a bit if icons present
          label.style.marginTop = '8px';
        }

        label.textContent = t;
        cell.appendChild(label);
      }

      container.appendChild(cell);
    }

    // legend intentionally removed per user preference
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
