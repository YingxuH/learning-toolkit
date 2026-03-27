#!/usr/bin/env python3
"""Comprehensive UI Visual Regression Tests for AI Engineer Learning Toolkit.

Run: python3 tests/run_ui_tests.py [URL]
Requires: pip install playwright && python3 -m playwright install chromium

Covers: layout, typography, colors, spacing, dark mode, responsiveness,
chat panel, console panel, sidebar, search, highlights, tables, code blocks,
callouts, interview cards, war stories, study plan, chapter navigation,
language toggle, reading progress, z-index stacking, overflow, accessibility.
"""

import sys
import re

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Install: pip install playwright && python -m playwright install chromium")
    sys.exit(1)

URL = sys.argv[1] if len(sys.argv) > 1 else "https://yingxuh.github.io/learning-toolkit/"

passed = 0
failed = 0
failures = []

def test(condition, name, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  \033[32mPASS\033[0m {name}")
    else:
        failed += 1
        failures.append(f"{name}: {detail}")
        print(f"  \033[31mFAIL\033[0m {name} - {detail}")

def group(name):
    print(f"\n\033[34m=== {name} ===\033[0m")

def px(val):
    if not val: return 0
    m = re.search(r'([\d.]+)px', str(val))
    return float(m.group(1)) if m else 0

def is_opaque(bg):
    if not bg: return False
    if 'rgba(0, 0, 0, 0)' in bg or bg == 'transparent': return False
    return True

def rgb_tuple(color_str):
    m = re.findall(r'(\d+)', color_str or '')
    return tuple(int(x) for x in m[:3]) if len(m) >= 3 else (0,0,0)

def luminance(rgb):
    r, g, b = [x/255 for x in rgb]
    return 0.2126*r + 0.7152*g + 0.0722*b

def contrast_ratio(fg, bg):
    l1 = luminance(rgb_tuple(fg)) + 0.05
    l2 = luminance(rgb_tuple(bg)) + 0.05
    return max(l1,l2) / min(l1,l2)

print(f"\nUI Tests for: {URL}\n")

with sync_playwright() as p:
    browser = p.chromium.launch()

    # ==========================================================
    # DESKTOP (1280x900)
    # ==========================================================
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    page.goto(URL, wait_until="networkidle")
    page.wait_for_timeout(3000)

    # --- Nav Bar ---
    group("Nav Bar - Layout")
    nav = page.evaluate('''() => {
        const s = getComputedStyle(document.getElementById('top-nav'));
        return {h: s.height, pos: s.position, bg: s.backgroundColor, z: s.zIndex,
                backdropFilter: s.backdropFilter || s.webkitBackdropFilter || '',
                borderBottom: s.borderBottomWidth, display: s.display, gap: s.gap};
    }''')
    test(px(nav['h']) >= 48 and px(nav['h']) <= 64, "Nav height 48-64px", f"got {nav['h']}")
    test(nav['pos'] == 'fixed', "Nav fixed position")
    test(int(nav['z']) >= 40, "Nav z-index >= 40", f"got {nav['z']}")
    test('blur' in nav['backdropFilter'], "Nav has backdrop-blur", f"got '{nav['backdropFilter']}'")
    test(px(nav['borderBottom']) >= 1, "Nav has bottom border", f"got {nav['borderBottom']}")
    test(nav['display'] == 'flex', "Nav uses flexbox")

    # --- Nav Buttons ---
    group("Nav Buttons - Sizes & Spacing")
    btns = page.evaluate('''() => {
        return Array.from(document.querySelectorAll('#top-nav button')).map(b => {
            const s = getComputedStyle(b);
            const r = b.getBoundingClientRect();
            return {id: b.id||b.title||b.textContent.trim(), w: r.width, h: r.height,
                    cursor: s.cursor, borderRadius: s.borderRadius};
        });
    }''')
    for btn in btns:
        name = btn['id']
        # Skip hidden buttons (auth-signout is hidden when not logged in)
        if btn['w'] == 0 and btn['h'] == 0:
            continue
        if name in ['Sign In', 'Out', 'auth-signin', 'auth-signout']:
            test(btn['h'] >= 28, f"'{name}' height >= 28px", f"got {btn['h']:.0f}px")
        else:
            test(min(btn['w'], btn['h']) >= 40, f"'{name}' min dim >= 40px", f"got {btn['w']:.0f}x{btn['h']:.0f}")
        test(px(btn['borderRadius']) >= 4, f"'{name}' has rounded corners", f"got {btn['borderRadius']}")

    # Check no nav buttons overlap
    rects = page.evaluate('''() => {
        return Array.from(document.querySelectorAll('#top-nav button')).map(b => {
            const r = b.getBoundingClientRect();
            return {l: r.left, r: r.right, t: r.top, b: r.bottom, id: b.id||b.textContent.trim()};
        });
    }''')
    for i in range(len(rects)):
        for j in range(i+1, len(rects)):
            a, b_ = rects[i], rects[j]
            overlaps = not (a['r'] <= b_['l'] or b_['r'] <= a['l'] or a['b'] <= b_['t'] or b_['b'] <= a['t'])
            test(not overlaps, f"Buttons '{a['id']}' and '{b_['id']}' don't overlap")

    # --- Sidebar ---
    group("Sidebar - Desktop")
    sb = page.evaluate('''() => {
        const el = document.getElementById('sidebar');
        const s = getComputedStyle(el);
        return {w: s.width, bg: s.backgroundColor, borderR: s.borderRightWidth,
                overflow: s.overflowY, transition: s.transition};
    }''')
    test(200 <= px(sb['w']) <= 300, "Width 200-300px", f"got {sb['w']}")
    test(is_opaque(sb['bg']), "Opaque background", f"got {sb['bg']}")
    test(px(sb['borderR']) >= 1, "Has right border", f"got {sb['borderR']}")
    test(sb['overflow'] in ('auto', 'scroll', 'overlay'), "Scrollable", f"overflow-y: {sb['overflow']}")

    # Sidebar collapse/expand
    page.click('#sidebar-toggle')
    page.wait_for_timeout(300)
    collapsed = page.evaluate("() => document.getElementById('sidebar').classList.contains('collapsed')")
    test(collapsed, "Sidebar collapses on toggle")
    page.click('#sidebar-toggle')
    page.wait_for_timeout(300)
    expanded = page.evaluate("() => !document.getElementById('sidebar').classList.contains('collapsed')")
    test(expanded, "Sidebar expands on second toggle")

    # --- TOC ---
    group("TOC Items - Styling")
    toc = page.evaluate('''() => {
        const items = document.querySelectorAll('.toc-item');
        const parts = document.querySelectorAll('.toc-part-title');
        const first = items[0] ? getComputedStyle(items[0]) : {};
        const section = document.querySelector('.toc-section');
        const sectionS = section ? getComputedStyle(section) : {};
        return {
            count: items.length, partCount: parts.length,
            fontSize: first.fontSize, color: first.color, cursor: first.cursor,
            minHeight: first.minHeight, display: first.display,
            sectionFontSize: sectionS.fontSize,
            sectionPaddingLeft: sectionS.paddingLeft,
        };
    }''')
    test(toc['count'] >= 50, f"50+ TOC items", f"got {toc['count']}")
    test(toc['partCount'] >= 4, f"4+ part headers", f"got {toc['partCount']}")
    test(toc['cursor'] == 'pointer', "Pointer cursor")
    test(toc['display'] == 'flex', "Flex display for alignment")
    test(px(toc['fontSize']) >= 11 and px(toc['fontSize']) <= 15, "Font 11-15px", f"got {toc['fontSize']}")
    test(px(toc['sectionPaddingLeft']) > px(toc.get('paddingLeft', '0px') or '16px'), "Sub-sections indented")

    # --- Content Typography ---
    group("Content Typography")
    typo = page.evaluate('''() => {
        const cs = (sel) => {const el=document.querySelector(sel); return el?getComputedStyle(el):{}};
        return {
            maxW: cs('#textbook-content').maxWidth,
            h2Size: cs('.chapter h2').fontSize, h2Weight: cs('.chapter h2').fontWeight,
            h2Color: cs('.chapter h2').color, h2LineHeight: cs('.chapter h2').lineHeight,
            h3Size: cs('.chapter h3').fontSize, h3Weight: cs('.chapter h3').fontWeight,
            h4Size: cs('.chapter h4').fontSize,
            pSize: cs('.chapter p').fontSize, pColor: cs('.chapter p').color,
            pLineHeight: cs('.chapter p').lineHeight,
            liSize: cs('.chapter li') ? cs('.chapter li').fontSize : null,
            strongColor: cs('.chapter strong').color,
        };
    }''')
    test(px(typo['maxW']) >= 700, "Content max-width >= 700px", f"got {typo['maxW']}")
    test(px(typo['h2Size']) >= 28, "H2 >= 28px", f"got {typo['h2Size']}")
    test(int(typo['h2Weight']) >= 600, "H2 weight >= 600", f"got {typo['h2Weight']}")
    test(px(typo['h3Size']) >= 18, "H3 >= 18px", f"got {typo['h3Size']}")
    test(px(typo['h4Size']) >= 15, "H4 >= 15px", f"got {typo['h4Size']}")
    test(14 <= px(typo['pSize']) <= 17, "P font 14-17px", f"got {typo['pSize']}")
    test(px(typo['pLineHeight']) >= 22, "P line-height >= 22px", f"got {typo['pLineHeight']}")
    test(typo['h2Color'] != typo['pColor'], "H2 vs P different colors")
    test(typo['strongColor'] != typo['pColor'], "Strong vs P different colors")
    bodyBg = page.evaluate("() => getComputedStyle(document.body).backgroundColor")
    h2_cr = contrast_ratio(typo['h2Color'], bodyBg)
    p_cr = contrast_ratio(typo['pColor'], bodyBg)
    test(h2_cr >= 4.5, f"H2 contrast ratio >= 4.5:1", f"got {h2_cr:.1f}")
    test(p_cr >= 3, f"P contrast ratio >= 3:1", f"got {p_cr:.1f}")

    # --- Code Blocks ---
    group("Code Blocks")
    code = page.evaluate('''() => {
        const pre = document.querySelector('.chapter pre');
        const c = document.querySelector('.chapter pre code');
        const inline = document.querySelector('.chapter code:not(pre code)');
        if (!pre) return null;
        const ps = getComputedStyle(pre);
        return {
            preBg: ps.backgroundColor, preRadius: ps.borderRadius, prePad: ps.padding,
            preBorder: ps.borderWidth, preOverflow: ps.overflowX, preFont: ps.fontFamily,
            preFontSize: ps.fontSize,
            codeColor: c ? getComputedStyle(c).color : null,
            inlineBg: inline ? getComputedStyle(inline).backgroundColor : null,
            inlineRadius: inline ? getComputedStyle(inline).borderRadius : null,
            inlineColor: inline ? getComputedStyle(inline).color : null,
        };
    }''')
    if code:
        test(is_opaque(code['preBg']), "Pre has background", f"got {code['preBg']}")
        test(px(code['preRadius']) >= 4, "Pre rounded corners", f"got {code['preRadius']}")
        test(px(code['prePad']) >= 12, "Pre padding >= 12px", f"got {code['prePad']}")
        test(code['preOverflow'] in ('auto', 'scroll', 'overlay'), "Pre scrollable", f"got {code['preOverflow']}")
        test('mono' in (code['preFont']or'').lower() or 'JetBrains' in (code['preFont']or''), "Monospace font")
        test(11 <= px(code['preFontSize']) <= 15, "Code font 11-15px", f"got {code['preFontSize']}")
        if code['inlineBg']:
            test(is_opaque(code['inlineBg']), "Inline code has bg", f"got {code['inlineBg']}")
            test(px(code['inlineRadius']) >= 2, "Inline code rounded", f"got {code['inlineRadius']}")

    # --- Callout Variants ---
    group("Callout Variants")
    callouts = page.evaluate('''() => {
        const types = {};
        document.querySelectorAll('.callout').forEach(el => {
            const cls = el.classList.contains('warning') ? 'warning' : el.classList.contains('tip') ? 'tip' : 'default';
            if (!types[cls]) {
                const s = getComputedStyle(el);
                const title = el.querySelector('.callout-title');
                types[cls] = {bg: s.backgroundColor, radius: s.borderRadius, padding: s.padding,
                    borderLeft: s.borderLeftWidth, titleColor: title?getComputedStyle(title).color:null};
            }
        });
        return types;
    }''')
    for ctype, styles in callouts.items():
        test(is_opaque(styles['bg']), f"Callout '{ctype}' has bg", f"got {styles['bg']}")
        test(px(styles['radius']) >= 4, f"Callout '{ctype}' rounded", f"got {styles['radius']}")
        test(px(styles['padding']) >= 12, f"Callout '{ctype}' padded", f"got {styles['padding']}")
        if ctype in ('warning', 'tip'):
            test(px(styles['borderLeft']) >= 2, f"Callout '{ctype}' has left border", f"got {styles['borderLeft']}")

    # --- Tables ---
    group("Tables")
    table = page.evaluate('''() => {
        const t = document.querySelector('.chapter table');
        if (!t) return null;
        const th = t.querySelector('th');
        const td = t.querySelector('td');
        const ts = getComputedStyle(t);
        return {
            borderCollapse: ts.borderCollapse, width: ts.width, border: ts.borderWidth,
            thBg: th ? getComputedStyle(th).backgroundColor : null,
            thColor: th ? getComputedStyle(th).color : null,
            thPad: th ? getComputedStyle(th).padding : null,
            tdColor: td ? getComputedStyle(td).color : null,
            overflow: ts.overflow,
        };
    }''')
    if table:
        test(table['borderCollapse'] == 'collapse', "Table border-collapse")
        test(is_opaque(table['thBg']), "TH has background", f"got {table['thBg']}")
        test(px(table['thPad']) >= 8, "TH padding >= 8px", f"got {table['thPad']}")
        test(table['thColor'] != table['tdColor'], "TH vs TD different colors")

    # --- Interview Q ---
    group("Interview Questions")
    iq = page.evaluate('''() => {
        const el = document.querySelector('.interview-q');
        if (!el) return null;
        const s = getComputedStyle(el);
        const label = el.querySelector('.q-label');
        const qtext = el.querySelector('.q-text');
        const atext = el.querySelector('.a-text');
        return {
            bg: s.backgroundColor, radius: s.borderRadius, pad: s.padding, border: s.borderWidth,
            labelColor: label?getComputedStyle(label).color:null,
            labelSize: label?getComputedStyle(label).fontSize:null,
            qColor: qtext?getComputedStyle(qtext).color:null,
            qWeight: qtext?getComputedStyle(qtext).fontWeight:null,
            aColor: atext?getComputedStyle(atext).color:null,
            aSize: atext?getComputedStyle(atext).fontSize:null,
        };
    }''')
    if iq:
        test(is_opaque(iq['bg']), "IQ has bg")
        test(px(iq['radius']) >= 6, "IQ rounded >= 6px", f"got {iq['radius']}")
        test(px(iq['pad']) >= 16, "IQ padding >= 16px", f"got {iq['pad']}")
        test(int(iq['qWeight']) >= 600, "Question text bold", f"got {iq['qWeight']}")
        test(iq['qColor'] != iq['aColor'], "Q vs A different colors")
        test(px(iq['labelSize']) <= 13, "Label small text <= 13px", f"got {iq['labelSize']}")

    # --- Chapter Nav ---
    group("Chapter Navigation")
    chapnav = page.evaluate('''() => {
        const nav = document.querySelector('.chapter-nav');
        if (!nav) return null;
        const links = nav.querySelectorAll('a');
        const s = getComputedStyle(nav);
        const first = links[0] ? getComputedStyle(links[0]) : {};
        return {
            display: s.display, borderTop: s.borderTopWidth,
            linkCount: links.length,
            linkMinH: first.minHeight, linkBg: first.backgroundColor,
            linkRadius: first.borderRadius, linkBorder: first.borderWidth,
        };
    }''')
    if chapnav:
        test(chapnav['display'] == 'flex', "Chapter nav is flex")
        test(px(chapnav['borderTop']) >= 1, "Has top border separator")
        test(chapnav['linkCount'] >= 1, "Has nav links", f"got {chapnav['linkCount']}")
        test(px(chapnav['linkMinH']) >= 40, "Link min-height >= 40px", f"got {chapnav['linkMinH']}")
        test(is_opaque(chapnav['linkBg']), "Link has bg")
        test(px(chapnav['linkRadius']) >= 6, "Link rounded", f"got {chapnav['linkRadius']}")

    # --- Study Plan / Hero ---
    group("Study Plan Hero")
    hero = page.evaluate('''() => {
        const ring = document.querySelector('.progress-ring');
        const pct = document.querySelector('.progress-pct');
        const goals = document.querySelectorAll('.goal-item');
        const changelog = document.querySelectorAll('.changelog-item');
        return {
            ringExists: !!ring,
            ringBg: ring ? getComputedStyle(ring).backgroundColor : null,
            pctSize: pct ? getComputedStyle(pct).fontSize : null,
            goalCount: goals.length,
            goalMinH: goals[0] ? getComputedStyle(goals[0]).minHeight || getComputedStyle(goals[0]).height : null,
            changelogCount: changelog.length,
        };
    }''')
    test(hero['ringExists'], "Progress ring exists")
    test(is_opaque(hero['ringBg']), "Progress ring has bg", f"got {hero['ringBg']}")
    test(px(hero['pctSize']) >= 30, "Progress % text large", f"got {hero['pctSize']}")
    test(hero['goalCount'] >= 4, "4+ study goals", f"got {hero['goalCount']}")
    test(hero['changelogCount'] >= 2, "2+ changelog entries", f"got {hero['changelogCount']}")

    # --- Reading Progress Bar ---
    group("Reading Progress Bar")
    progress = page.evaluate('''() => {
        // There may be two elements with this ID (one in HTML, one created by app.js)
        const els = document.querySelectorAll('#reading-progress, [id="reading-progress"]');
        for (const el of els) {
            const s = getComputedStyle(el);
            if (s.position === 'fixed') return {pos: s.position, h: s.height, bg: s.backgroundColor, z: s.zIndex, top: s.top};
        }
        return null;
    }''')
    if progress:
        test(progress['pos'] == 'fixed', "Progress bar fixed")
        test(px(progress['h']) <= 4, "Height <= 4px", f"got {progress['h']}")
        z = 0
        try: z = int(progress['z'])
        except: pass
        test(z >= 30 or progress['z'] == 'auto', "z-index reasonable", f"got {progress['z']}")

    # --- Chapter Indicator ---
    group("Chapter Indicator")
    indicator = page.evaluate('''() => {
        const el = document.getElementById('chapter-indicator');
        if (!el) return null;
        const s = getComputedStyle(el);
        return {pos: s.position, pointerEvents: s.pointerEvents, z: s.zIndex, radius: s.borderRadius};
    }''')
    if indicator:
        test(indicator['pos'] == 'fixed', "Indicator fixed")
        test(indicator['pointerEvents'] == 'none', "Non-interactive (pointer-events:none)")
        test(px(indicator['radius']) >= 10, "Pill-shaped", f"got {indicator['radius']}")

    # ==========================================================
    # CHAT PANEL
    # ==========================================================
    group("Chat Panel - Full Audit")
    page.click('#chat-toggle')
    page.wait_for_timeout(500)

    chat = page.evaluate('''() => {
        const cs = (el) => el ? getComputedStyle(el) : {};
        const panel = document.getElementById('chat-panel');
        const header = panel.querySelector('.chat-header');
        const input = document.getElementById('chat-input');
        const send = document.getElementById('chat-send');
        const msgs = document.getElementById('chat-messages');
        const bubble = document.querySelector('.msg-bubble');
        const banner = document.getElementById('chat-context-banner');
        const clearBtn = document.getElementById('chat-clear');
        const closeBtn = document.getElementById('chat-close');
        return {
            display: cs(panel).display, bg: cs(panel).backgroundColor,
            borderL: cs(panel).borderLeftWidth, w: cs(panel).width,
            flexDir: cs(panel).flexDirection,
            headerBg: cs(header).backgroundColor, headerBorderB: cs(header).borderBottomWidth,
            inputBg: cs(input).backgroundColor, inputColor: cs(input).color,
            inputBorder: cs(input).borderWidth, inputRadius: cs(input).borderRadius,
            inputFont: cs(input).fontSize, inputResize: cs(input).resize,
            sendBg: cs(send).backgroundColor, sendW: cs(send).width, sendH: cs(send).height,
            sendRadius: cs(send).borderRadius, sendColor: cs(send).color,
            msgsOverflow: cs(msgs).overflowY, msgsFlex: cs(msgs).flex,
            bubbleBg: bubble?cs(bubble).backgroundColor:'NONE',
            bubbleColor: bubble?cs(bubble).color:'NONE',
            bubbleRadius: bubble?cs(bubble).borderRadius:'0',
            bubblePad: bubble?cs(bubble).padding:'0',
            bannerHidden: banner.classList.contains('hidden'),
            clearExists: !!clearBtn, closeExists: !!closeBtn,
        };
    }''')
    test(chat['display'] != 'none', "Visible after toggle")
    test(is_opaque(chat['bg']), "Panel has bg", f"got {chat['bg']}")
    test(px(chat['borderL']) >= 1, "Has left border", f"got {chat['borderL']}")
    test(px(chat['w']) >= 300, "Width >= 300px", f"got {chat['w']}")
    test(chat['flexDir'] == 'column', "Column flex layout")
    test(px(chat['headerBorderB']) >= 1, "Header has bottom border")
    test(is_opaque(chat['inputBg']), "Input has bg", f"got {chat['inputBg']}")
    test(chat['inputColor'] != 'rgba(0, 0, 0, 0)', "Input has text color")
    test(px(chat['inputBorder']) >= 1, "Input has border", f"got {chat['inputBorder']}")
    test(px(chat['inputRadius']) >= 4, "Input rounded", f"got {chat['inputRadius']}")
    test(13 <= px(chat['inputFont']) <= 16, "Input font 13-16px", f"got {chat['inputFont']}")
    test(is_opaque(chat['sendBg']), "Send btn has bg", f"got {chat['sendBg']}")
    test(px(chat['sendW']) >= 32, "Send btn width >= 32px")
    test(px(chat['sendRadius']) >= 4, "Send btn rounded")
    test(chat['msgsOverflow'] in ('auto','scroll','overlay'), "Messages scrollable")
    if chat['bubbleBg'] != 'NONE':
        test(is_opaque(chat['bubbleBg']), "Bubble has bg", f"got {chat['bubbleBg']}")
        test(chat['bubbleColor'] != 'rgba(0, 0, 0, 0)', "Bubble has color")
        test(px(chat['bubbleRadius']) >= 6, "Bubble rounded", f"got {chat['bubbleRadius']}")
        test(px(chat['bubblePad']) >= 8, "Bubble padded", f"got {chat['bubblePad']}")
    test(chat['bannerHidden'], "Context banner hidden by default")
    test(chat['clearExists'], "Clear button exists")
    test(chat['closeExists'], "Close button exists")

    # Close chat
    page.click('#chat-close')
    page.wait_for_timeout(300)
    test(page.evaluate("() => document.getElementById('chat-panel').classList.contains('hidden')"), "Chat closes on X click")

    # ==========================================================
    # CONSOLE PANEL
    # ==========================================================
    group("Console Panel - Full Audit")
    page.click('#console-toggle')
    page.wait_for_timeout(500)

    con = page.evaluate('''() => {
        const cs = (el) => el ? getComputedStyle(el) : {};
        const panel = document.getElementById('console-panel');
        const content = document.getElementById('console-content');
        const cards = content.querySelectorAll('[class*="rounded-lg"]');
        const btns = content.querySelectorAll('button');
        return {
            display: cs(panel).display, bg: cs(panel).backgroundColor,
            borderL: cs(panel).borderLeftWidth, w: cs(panel).width,
            overflow: cs(panel).overflowY,
            childCount: content.children.length,
            cardCount: cards.length,
            btnCount: btns.length,
            hasResetAll: Array.from(btns).some(b => b.textContent.includes('Reset ALL')),
            hasExport: Array.from(btns).some(b => b.textContent.includes('Export')),
            hasSync: Array.from(btns).some(b => b.textContent.includes('Sync')),
        };
    }''')
    test(con['display'] != 'none', "Visible after toggle")
    test(is_opaque(con['bg']), "Has bg", f"got {con['bg']}")
    test(px(con['borderL']) >= 1, "Has left border")
    test(px(con['w']) >= 300, "Width >= 300px", f"got {con['w']}")
    test(con['overflow'] in ('auto','scroll','overlay'), "Scrollable")
    test(con['childCount'] >= 3, "Has sections", f"got {con['childCount']}")
    test(con['cardCount'] >= 4, "Has stat cards", f"got {con['cardCount']}")
    test(con['btnCount'] >= 4, "Has action buttons", f"got {con['btnCount']}")
    test(con['hasResetAll'], "Has 'Reset ALL' button")
    test(con['hasExport'], "Has 'Export' button")
    test(con['hasSync'], "Has 'Sync' button")

    page.click('#console-close')
    page.wait_for_timeout(300)
    test(page.evaluate("() => document.getElementById('console-panel').classList.contains('hidden')"), "Console closes on X click")

    # Chat and Console mutual exclusivity
    group("Panel Mutual Exclusivity")
    page.click('#chat-toggle')
    page.wait_for_timeout(300)
    page.click('#console-toggle')
    page.wait_for_timeout(300)
    test(page.evaluate("() => document.getElementById('chat-panel').classList.contains('hidden')"), "Chat closes when console opens")

    page.click('#console-close')
    page.wait_for_timeout(300)

    # ==========================================================
    # DARK MODE - Comprehensive
    # ==========================================================
    group("Dark Mode - Comprehensive")
    page.click('#theme-toggle')
    page.wait_for_timeout(500)

    dark = page.evaluate('''() => {
        const cs = (sel) => {const el=document.querySelector(sel); return el?getComputedStyle(el):{}};
        const isDark = document.documentElement.classList.contains('dark');
        return {
            isDark,
            bodyBg: cs('body').backgroundColor,
            navBg: cs('#top-nav').backgroundColor,
            sidebarBg: cs('#sidebar').backgroundColor,
            h2Color: cs('.chapter h2').color,
            pColor: cs('.chapter p').color,
            preBg: cs('.chapter pre').backgroundColor,
            preCodeColor: cs('.chapter pre code') ? cs('.chapter pre code').color : null,
            calloutBg: document.querySelector('.callout') ? cs('.callout').backgroundColor : null,
            iqBg: document.querySelector('.interview-q') ? cs('.interview-q').backgroundColor : null,
            tocColor: cs('.toc-item').color,
            inlineCodeBg: document.querySelector('.chapter code:not(pre code)') ? cs('.chapter code:not(pre code)').backgroundColor : null,
            tableBorder: document.querySelector('.chapter table') ? cs('.chapter table').borderColor : null,
            thBg: document.querySelector('.chapter th') ? cs('.chapter th').backgroundColor : null,
        };
    }''')
    test(dark['isDark'], "Dark class applied")
    # All backgrounds should be dark
    for key in ['bodyBg', 'sidebarBg', 'preBg']:
        rgb = rgb_tuple(dark[key])
        test(luminance(rgb) < 0.15, f"Dark {key} is dark", f"got {dark[key]}")
    # All text should be light
    for key in ['h2Color', 'pColor']:
        rgb = rgb_tuple(dark[key])
        test(luminance(rgb) > 0.5, f"Dark {key} is light", f"got {dark[key]}")
    # Contrast checks
    test(contrast_ratio(dark['h2Color'], dark['bodyBg']) >= 4.5, "Dark H2 contrast >= 4.5:1")
    test(contrast_ratio(dark['pColor'], dark['bodyBg']) >= 3, "Dark P contrast >= 3:1")
    if dark['calloutBg']:
        test(is_opaque(dark['calloutBg']), "Dark callout has bg")
    if dark['iqBg']:
        test(is_opaque(dark['iqBg']), "Dark IQ has bg")
    if dark['inlineCodeBg']:
        test(is_opaque(dark['inlineCodeBg']), "Dark inline code has bg")

    # Dark mode chat panel
    page.click('#chat-toggle')
    page.wait_for_timeout(500)
    dark_chat = page.evaluate('''() => {
        const cs = (el) => el ? getComputedStyle(el) : {};
        const panel = document.getElementById('chat-panel');
        const input = document.getElementById('chat-input');
        const send = document.getElementById('chat-send');
        const bubble = document.querySelector('.chat-message.assistant .msg-bubble');
        return {
            panelBg: cs(panel).backgroundColor,
            inputBg: cs(input).backgroundColor, inputColor: cs(input).color,
            sendBg: cs(send).backgroundColor,
            bubbleBg: bubble ? cs(bubble).backgroundColor : 'NONE',
            bubbleColor: bubble ? cs(bubble).color : 'NONE',
        };
    }''')
    test(is_opaque(dark_chat['panelBg']), "Dark chat panel has bg", f"got {dark_chat['panelBg']}")
    test(luminance(rgb_tuple(dark_chat['panelBg'])) < 0.15, "Dark chat bg is dark")
    test(is_opaque(dark_chat['inputBg']), "Dark chat input has bg", f"got {dark_chat['inputBg']}")
    test(is_opaque(dark_chat['sendBg']), "Dark send btn has bg", f"got {dark_chat['sendBg']}")
    if dark_chat['bubbleBg'] != 'NONE':
        test(is_opaque(dark_chat['bubbleBg']), "Dark bubble has bg", f"got {dark_chat['bubbleBg']}")
        test(luminance(rgb_tuple(dark_chat['bubbleColor'])) > 0.5, "Dark bubble text is light")

    page.click('#chat-close')
    page.wait_for_timeout(200)
    # Switch back to light
    page.click('#theme-toggle')
    page.wait_for_timeout(300)

    # ==========================================================
    # LANGUAGE TOGGLE
    # ==========================================================
    group("Language Toggle")
    lang_btn = page.evaluate("() => document.getElementById('lang-toggle').textContent.trim()")
    test(lang_btn == 'EN', "Starts as EN", f"got '{lang_btn}'")
    page.click('#lang-toggle')
    page.wait_for_timeout(1000)
    zh_title = page.evaluate("() => document.querySelector('.nav-title').textContent")
    zh_btn = page.evaluate("() => document.getElementById('lang-toggle').textContent.trim()")
    test('工程师' in zh_title or '手册' in zh_title, "Title changes to Chinese", f"got '{zh_title}'")
    test(zh_btn == '中', "Button shows 中", f"got '{zh_btn}'")
    # Switch back
    page.click('#lang-toggle')
    page.wait_for_timeout(500)

    # ==========================================================
    # SEARCH INTERACTION
    # ==========================================================
    group("Search Interaction")
    page.fill('#global-search', 'attention')
    page.wait_for_timeout(500)
    search_res = page.evaluate('''() => {
        const dropdown = document.getElementById('search-results');
        const items = dropdown.querySelectorAll('.search-result-item');
        return {
            visible: !dropdown.classList.contains('hidden'),
            count: items.length,
            hasChapter: items[0] ? !!items[0].querySelector('.search-result-chapter') : false,
            hasText: items[0] ? !!items[0].querySelector('.search-result-text') : false,
        };
    }''')
    test(search_res['visible'], "Dropdown appears on search")
    test(search_res['count'] > 0, f"Results found", f"got {search_res['count']}")
    test(search_res['hasChapter'], "Result has chapter label")
    test(search_res['hasText'], "Result has snippet text")

    # Clear search
    page.fill('#global-search', '')
    page.wait_for_timeout(300)
    hidden = page.evaluate("() => document.getElementById('search-results').classList.contains('hidden')")
    test(hidden, "Dropdown hides on empty query")

    # ==========================================================
    # Z-INDEX STACKING
    # ==========================================================
    group("Z-Index Stacking Order")
    zstack = page.evaluate('''() => {
        const z = (id) => {
            const el = document.getElementById(id);
            if (!el) return 0;
            const v = getComputedStyle(el).zIndex;
            return v === 'auto' ? 0 : parseInt(v) || 0;
        };
        return {
            nav: z('top-nav'),
            toolbar: z('highlight-toolbar'),
            indicator: z('chapter-indicator'),
        };
    }''')
    test(zstack['toolbar'] > zstack['nav'], "Toolbar above nav", f"toolbar={zstack['toolbar']}, nav={zstack['nav']}")
    test(zstack['nav'] >= 40, "Nav z-index >= 40", f"got {zstack['nav']}")

    # ==========================================================
    # NO HORIZONTAL OVERFLOW
    # ==========================================================
    group("No Horizontal Overflow")
    overflow = page.evaluate('''() => {
        return {
            bodyScrollW: document.body.scrollWidth,
            viewportW: window.innerWidth,
            hasHScroll: document.body.scrollWidth > window.innerWidth,
        };
    }''')
    test(not overflow['hasHScroll'], "No horizontal scroll", f"body={overflow['bodyScrollW']}px, viewport={overflow['viewportW']}px")

    page.close()

    # ==========================================================
    # MOBILE (390x844)
    # ==========================================================
    page = browser.new_page(viewport={"width": 390, "height": 844})
    page.goto(URL, wait_until="networkidle")
    page.wait_for_timeout(3000)

    group("Mobile - Layout")
    mob = page.evaluate('''() => {
        const cs = (el) => el ? getComputedStyle(el) : {};
        return {
            navH: cs(document.getElementById('top-nav')).height,
            sidebarCollapsed: document.getElementById('sidebar').classList.contains('collapsed'),
            contentW: parseFloat(cs(document.getElementById('content-area')).width),
            viewportW: window.innerWidth,
            noHScroll: document.body.scrollWidth <= window.innerWidth,
        };
    }''')
    test(px(mob['navH']) >= 48, "Nav height >= 48px")
    test(mob['sidebarCollapsed'], "Sidebar collapsed by default")
    test(mob['contentW'] <= mob['viewportW'], "Content fits viewport")
    test(mob['noHScroll'], "No horizontal scroll on mobile")

    group("Mobile - Sidebar")
    page.click('#sidebar-toggle')
    page.wait_for_timeout(500)
    msb = page.evaluate('''() => {
        const sb = document.getElementById('sidebar');
        const ol = document.getElementById('sidebar-overlay');
        const cl = document.getElementById('sidebar-close');
        const cs = (el) => el ? getComputedStyle(el) : {};
        return {
            open: !sb.classList.contains('collapsed'),
            pos: cs(sb).position, w: cs(sb).width, z: cs(sb).zIndex,
            overlayVis: ol ? cs(ol).display !== 'none' : false,
            closeVis: cl ? cs(cl).display !== 'none' : false,
        };
    }''')
    test(msb['open'], "Opens on toggle")
    test(msb['pos'] == 'fixed', "Fixed position")
    test(px(msb['w']) >= 250, "Width >= 250px", f"got {msb['w']}")
    test(int(msb['z']) >= 30, "z-index >= 30")
    # Close via overlay
    page.click('#sidebar-toggle')
    page.wait_for_timeout(300)

    group("Mobile - Chat Panel")
    page.click('#chat-toggle')
    page.wait_for_timeout(500)
    mchat = page.evaluate('''() => {
        const panel = document.getElementById('chat-panel');
        const cs = getComputedStyle(panel);
        return {w: cs.width, pos: cs.position, z: cs.zIndex, bg: cs.backgroundColor};
    }''')
    test(px(mchat['w']) >= 380, "Full-width on mobile", f"got {mchat['w']}")
    test(mchat['pos'] == 'fixed', "Fixed position")
    test(is_opaque(mchat['bg']), "Has bg on mobile", f"got {mchat['bg']}")
    page.click('#chat-close')
    page.wait_for_timeout(300)

    group("Mobile - Console Panel")
    page.click('#console-toggle')
    page.wait_for_timeout(500)
    mcon = page.evaluate('''() => {
        const panel = document.getElementById('console-panel');
        const cs = getComputedStyle(panel);
        return {w: cs.width, pos: cs.position, bg: cs.backgroundColor};
    }''')
    test(px(mcon['w']) >= 380, "Full-width on mobile", f"got {mcon['w']}")
    test(is_opaque(mcon['bg']), "Has bg on mobile")
    page.click('#console-close')
    page.wait_for_timeout(300)

    group("Mobile - Typography")
    mtypo = page.evaluate('''() => {
        const cs = (sel) => {const el=document.querySelector(sel); return el?getComputedStyle(el):{}};
        return {
            pSize: cs('.chapter p').fontSize,
            pLineH: cs('.chapter p').lineHeight,
            h2Size: cs('.chapter h2').fontSize,
            preSize: document.querySelector('.chapter pre') ? cs('.chapter pre').fontSize : null,
        };
    }''')
    test(px(mtypo['pSize']) >= 14, "Mobile P font >= 14px", f"got {mtypo['pSize']}")
    test(px(mtypo['pLineH']) >= 20, "Mobile P line-height >= 20px", f"got {mtypo['pLineH']}")
    test(px(mtypo['h2Size']) >= 22, "Mobile H2 >= 22px", f"got {mtypo['h2Size']}")

    # ==========================================================
    # TABLET (768x1024)
    # ==========================================================
    page.close()
    page = browser.new_page(viewport={"width": 768, "height": 1024})
    page.goto(URL, wait_until="networkidle")
    page.wait_for_timeout(3000)

    group("Tablet Layout (768px)")
    tablet = page.evaluate('''() => {
        const sb = document.getElementById('sidebar');
        const sbCollapsed = sb.classList.contains('collapsed');
        const sbPos = getComputedStyle(sb).position;
        return {
            noHScroll: document.body.scrollWidth <= window.innerWidth,
            contentW: parseFloat(getComputedStyle(document.getElementById('content-area')).width),
            sidebarCollapsed: sbCollapsed,
            sidebarIsOverlay: sbPos === 'fixed',
        };
    }''')
    test(tablet['noHScroll'], "No horizontal scroll on tablet")
    test(tablet['sidebarCollapsed'] or tablet['sidebarIsOverlay'], "Sidebar collapsed or overlay on tablet", f"collapsed={tablet['sidebarCollapsed']}, overlay={tablet['sidebarIsOverlay']}")
    test(tablet['contentW'] <= 768, "Content fits tablet viewport", f"got {tablet['contentW']:.0f}px")

    page.close()
    browser.close()

# ==========================================================
# SUMMARY
# ==========================================================
print(f"\n{'='*50}")
total = passed + failed
if failed == 0:
    print(f"\033[32mALL TESTS PASSED: {passed}/{total}\033[0m")
else:
    print(f"\033[31mSOME TESTS FAILED: {passed}/{total} passed, {failed} failed\033[0m")
    print("\nFailures:")
    for f in failures:
        print(f"  - {f}")

sys.exit(0 if failed == 0 else 1)
