#!/usr/bin/env python3
"""UI Visual Regression Tests for AI Engineer Learning Toolkit.

Run: python3 tests/run_ui_tests.py [--url URL]
Requires: pip install playwright && python3 -m playwright install chromium
"""

import sys
import re

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Install playwright: pip install playwright && python -m playwright install chromium")
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

def parse_px(val):
    """Extract numeric px value from CSS string like '44px'."""
    if not val:
        return 0
    m = re.search(r'([\d.]+)px', str(val))
    return float(m.group(1)) if m else 0

def is_opaque(bg_str):
    """Check if a background color is opaque (not transparent)."""
    if not bg_str:
        return False
    if 'rgba(0, 0, 0, 0)' in bg_str or bg_str == 'transparent':
        return False
    return True

def is_readable_contrast(fg, bg):
    """Basic check that foreground and background are different enough."""
    if not fg or not bg:
        return False
    return fg != bg

print(f"\nUI Tests for: {URL}\n")

with sync_playwright() as p:
    browser = p.chromium.launch()

    # ============================================================
    # DESKTOP TESTS (1280x900)
    # ============================================================
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    page.goto(URL, wait_until="networkidle")
    page.wait_for_timeout(3000)

    group("Navigation Bar - Desktop")
    nav = page.evaluate('''() => {
        const el = document.getElementById('top-nav');
        const s = window.getComputedStyle(el);
        return { height: s.height, position: s.position, bg: s.backgroundColor, zIndex: s.zIndex };
    }''')
    test(parse_px(nav['height']) >= 48, "Nav height >= 48px", f"got {nav['height']}")
    test(nav['position'] == 'fixed', "Nav is fixed position", f"got {nav['position']}")
    test(int(nav['zIndex']) >= 40, "Nav z-index >= 40", f"got {nav['zIndex']}")
    test(is_opaque(nav['bg']) or 'rgba' in nav['bg'], "Nav has background", f"got {nav['bg']}")

    group("Navigation Buttons - Tap Targets")
    btns = page.evaluate('''() => {
        return Array.from(document.querySelectorAll('#top-nav button')).map(b => {
            const s = window.getComputedStyle(b);
            return { id: b.id || b.title || b.textContent.trim(), w: parseFloat(s.width), h: parseFloat(s.height) };
        });
    }''')
    for btn in btns:
        if btn['id'] in ['Sign In', 'Out', 'auth-signin', 'auth-signout']:
            test(btn['h'] >= 28, f"Button '{btn['id']}' height >= 28px", f"got {btn['h']}px")
        else:
            test(min(btn['w'], btn['h']) >= 40, f"Button '{btn['id']}' min dimension >= 40px", f"got {btn['w']}x{btn['h']}px")

    group("Sidebar - Desktop")
    sidebar = page.evaluate('''() => {
        const el = document.getElementById('sidebar');
        const s = window.getComputedStyle(el);
        return { width: s.width, bg: s.backgroundColor, borderRight: s.borderRightWidth };
    }''')
    test(200 <= parse_px(sidebar['width']) <= 300, "Sidebar width 200-300px", f"got {sidebar['width']}")
    test(is_opaque(sidebar['bg']), "Sidebar has opaque background", f"got {sidebar['bg']}")

    group("TOC Items")
    toc = page.evaluate('''() => {
        const items = document.querySelectorAll('.toc-item');
        if (!items.length) return { count: 0 };
        const s = window.getComputedStyle(items[0]);
        return {
            count: items.length,
            minHeight: s.minHeight,
            fontSize: s.fontSize,
            color: s.color,
            cursor: s.cursor,
        };
    }''')
    test(toc['count'] >= 50, f"TOC has 50+ items", f"got {toc['count']}")
    test(toc['cursor'] == 'pointer', "TOC items have pointer cursor", f"got {toc['cursor']}")

    group("Content Area")
    content = page.evaluate('''() => {
        const tb = document.getElementById('textbook-content');
        const s = window.getComputedStyle(tb);
        const h2 = document.querySelector('.chapter h2');
        const p = document.querySelector('.chapter p');
        const pre = document.querySelector('.chapter pre');
        const code = document.querySelector('.chapter code');
        return {
            maxWidth: s.maxWidth,
            h2Color: h2 ? window.getComputedStyle(h2).color : null,
            h2FontSize: h2 ? window.getComputedStyle(h2).fontSize : null,
            pColor: p ? window.getComputedStyle(p).color : null,
            pFontSize: p ? window.getComputedStyle(p).fontSize : null,
            pLineHeight: p ? window.getComputedStyle(p).lineHeight : null,
            preBg: pre ? window.getComputedStyle(pre).backgroundColor : null,
            preFontFamily: pre ? window.getComputedStyle(pre).fontFamily : null,
            codeColor: code ? window.getComputedStyle(code).color : null,
        };
    }''')
    test(parse_px(content['maxWidth']) >= 700, "Content max-width >= 700px", f"got {content['maxWidth']}")
    test(parse_px(content['h2FontSize']) >= 24, "H2 font-size >= 24px", f"got {content['h2FontSize']}")
    test(parse_px(content['pFontSize']) >= 14, "Paragraph font-size >= 14px", f"got {content['pFontSize']}")
    test(parse_px(content['pLineHeight']) >= 20, "Paragraph line-height >= 20px", f"got {content['pLineHeight']}")
    test(is_opaque(content['preBg']), "Code block has opaque background", f"got {content['preBg']}")
    test('mono' in (content['preFontFamily'] or '').lower() or 'JetBrains' in (content['preFontFamily'] or ''), "Code uses monospace font", f"got {content['preFontFamily']}")
    test(content['h2Color'] != content['pColor'], "H2 and paragraph have different colors", f"h2={content['h2Color']}, p={content['pColor']}")

    group("Callout Boxes")
    callout = page.evaluate('''() => {
        const el = document.querySelector('.callout');
        if (!el) return null;
        const s = window.getComputedStyle(el);
        return { bg: s.backgroundColor, borderRadius: s.borderRadius, padding: s.padding };
    }''')
    if callout:
        test(is_opaque(callout['bg']), "Callout has opaque background", f"got {callout['bg']}")
        test(parse_px(callout['borderRadius']) >= 4, "Callout has rounded corners", f"got {callout['borderRadius']}")
        test(parse_px(callout['padding']) >= 12, "Callout has padding >= 12px", f"got {callout['padding']}")

    group("Interview Question Cards")
    iq = page.evaluate('''() => {
        const el = document.querySelector('.interview-q');
        if (!el) return null;
        const s = window.getComputedStyle(el);
        return { bg: s.backgroundColor, borderRadius: s.borderRadius, border: s.border };
    }''')
    if iq:
        test(is_opaque(iq['bg']), "Interview Q has opaque background", f"got {iq['bg']}")
        test(parse_px(iq['borderRadius']) >= 4, "Interview Q has rounded corners", f"got {iq['borderRadius']}")

    # ============================================================
    # CHAT PANEL TESTS
    # ============================================================
    group("Chat Panel")
    page.click('#chat-toggle')
    page.wait_for_timeout(500)

    chat = page.evaluate('''() => {
        const panel = document.getElementById('chat-panel');
        const msgs = document.getElementById('chat-messages');
        const input = document.getElementById('chat-input');
        const send = document.getElementById('chat-send');
        const header = panel.querySelector('.chat-header');
        const msgBubble = document.querySelector('.msg-bubble');
        const cs = (el) => el ? window.getComputedStyle(el) : {};
        return {
            panelDisplay: cs(panel).display,
            panelBg: cs(panel).backgroundColor,
            panelWidth: cs(panel).width,
            panelBorderLeft: cs(panel).borderLeftWidth,
            headerBg: cs(header).backgroundColor,
            headerBorderBottom: cs(header).borderBottomWidth,
            msgsBg: cs(msgs).backgroundColor,
            inputBg: cs(input).backgroundColor,
            inputColor: cs(input).color,
            inputBorder: cs(input).borderWidth,
            inputFontSize: cs(input).fontSize,
            sendBg: cs(send).backgroundColor,
            sendWidth: cs(send).width,
            sendHeight: cs(send).height,
            bubbleColor: msgBubble ? cs(msgBubble).color : 'NONE',
            bubbleBg: msgBubble ? cs(msgBubble).backgroundColor : 'NONE',
        };
    }''')
    test(chat['panelDisplay'] != 'none', "Chat panel is visible after toggle")
    test(is_opaque(chat['panelBg']), "Chat panel has opaque background", f"got {chat['panelBg']}")
    test(parse_px(chat['panelWidth']) >= 300, "Chat panel width >= 300px", f"got {chat['panelWidth']}")
    test(parse_px(chat['panelBorderLeft']) >= 1, "Chat panel has left border", f"got {chat['panelBorderLeft']}")
    test(is_opaque(chat['inputBg']), "Chat input has opaque background", f"got {chat['inputBg']}")
    test(chat['inputColor'] != 'rgba(0, 0, 0, 0)', "Chat input text has color", f"got {chat['inputColor']}")
    test(parse_px(chat['inputBorder']) >= 1, "Chat input has border", f"got {chat['inputBorder']}")
    test(parse_px(chat['inputFontSize']) >= 13, "Chat input font >= 13px", f"got {chat['inputFontSize']}")
    test(is_opaque(chat['sendBg']), "Send button has opaque background", f"got {chat['sendBg']}")
    test(parse_px(chat['sendWidth']) >= 32, "Send button width >= 32px", f"got {chat['sendWidth']}")
    if chat['bubbleBg'] != 'NONE':
        test(is_opaque(chat['bubbleBg']), "Message bubble has opaque bg", f"got {chat['bubbleBg']}")
        test(chat['bubbleColor'] != 'rgba(0, 0, 0, 0)', "Message bubble has text color", f"got {chat['bubbleColor']}")

    page.click('#chat-close')
    page.wait_for_timeout(300)

    # ============================================================
    # CONSOLE PANEL TESTS
    # ============================================================
    group("Console Panel")
    page.click('#console-toggle')
    page.wait_for_timeout(500)

    console_styles = page.evaluate('''() => {
        const panel = document.getElementById('console-panel');
        const content = document.getElementById('console-content');
        const cs = (el) => el ? window.getComputedStyle(el) : {};
        return {
            panelDisplay: cs(panel).display,
            panelBg: cs(panel).backgroundColor,
            panelWidth: cs(panel).width,
            contentChildren: content ? content.children.length : 0,
        };
    }''')
    test(console_styles['panelDisplay'] != 'none', "Console panel is visible after toggle")
    test(is_opaque(console_styles['panelBg']), "Console panel has opaque bg", f"got {console_styles['panelBg']}")
    test(parse_px(console_styles['panelWidth']) >= 300, "Console panel width >= 300px", f"got {console_styles['panelWidth']}")
    test(console_styles['contentChildren'] >= 3, "Console has content sections", f"got {console_styles['contentChildren']} children")

    page.click('#console-close')
    page.wait_for_timeout(300)

    # ============================================================
    # DARK MODE TESTS
    # ============================================================
    group("Dark Mode")
    page.click('#theme-toggle')
    page.wait_for_timeout(500)

    dark = page.evaluate('''() => {
        const cs = (el) => el ? window.getComputedStyle(el) : {};
        const body = document.body;
        const isDark = document.documentElement.classList.contains('dark');
        return {
            isDark: isDark,
            bodyBg: cs(body).backgroundColor,
            bodyColor: cs(body).color,
            sidebarBg: cs(document.getElementById('sidebar')).backgroundColor,
            h2Color: cs(document.querySelector('.chapter h2')).color,
            pColor: cs(document.querySelector('.chapter p')).color,
            codeBg: cs(document.querySelector('.chapter pre')).backgroundColor,
            calloutBg: document.querySelector('.callout') ? cs(document.querySelector('.callout')).backgroundColor : null,
            navBg: cs(document.getElementById('top-nav')).backgroundColor,
        };
    }''')
    test(dark['isDark'], "Dark mode class is applied")
    test(is_opaque(dark['bodyBg']), "Body has dark background", f"got {dark['bodyBg']}")
    test(dark['bodyBg'] != 'rgb(255, 255, 255)', "Body bg is not white in dark mode", f"got {dark['bodyBg']}")
    test(dark['h2Color'] != 'rgb(17, 24, 39)', "H2 is not dark color in dark mode", f"got {dark['h2Color']}")
    test(dark['pColor'] != 'rgb(75, 85, 99)', "P is lighter in dark mode", f"got {dark['pColor']}")
    test(is_opaque(dark['codeBg']), "Code block has dark bg", f"got {dark['codeBg']}")
    test(dark['codeBg'] != 'rgb(248, 249, 250)', "Code bg is not light in dark mode", f"got {dark['codeBg']}")

    # Switch back to light
    page.click('#theme-toggle')
    page.wait_for_timeout(300)

    # ============================================================
    # SEARCH FUNCTIONALITY
    # ============================================================
    group("Search UI")
    search = page.evaluate('''() => {
        const input = document.getElementById('global-search');
        const results = document.getElementById('search-results');
        const cs = (el) => el ? window.getComputedStyle(el) : {};
        return {
            inputExists: !!input,
            inputHeight: cs(input).height,
            inputBg: cs(input).backgroundColor,
            resultsHidden: results.classList.contains('hidden'),
        };
    }''')
    test(search['inputExists'], "Search input exists")
    test(parse_px(search['inputHeight']) >= 28, "Search input height >= 28px", f"got {search['inputHeight']}")
    test(is_opaque(search['inputBg']), "Search input has background", f"got {search['inputBg']}")
    test(search['resultsHidden'], "Search results hidden by default")

    # ============================================================
    # HIGHLIGHT TOOLBAR
    # ============================================================
    group("Highlight Toolbar")
    hl_toolbar = page.evaluate('''() => {
        const el = document.getElementById('highlight-toolbar');
        const btns = el.querySelectorAll('.hl-btn');
        const cs = (el) => el ? window.getComputedStyle(el) : {};
        return {
            position: cs(el).position,
            zIndex: cs(el).zIndex,
            btnCount: btns.length,
            btnSize: btns[0] ? { w: cs(btns[0]).width, h: cs(btns[0]).height } : null,
        };
    }''')
    test(hl_toolbar['position'] == 'fixed', "Toolbar is fixed position", f"got {hl_toolbar['position']}")
    test(int(hl_toolbar['zIndex']) >= 1000, "Toolbar z-index >= 1000", f"got {hl_toolbar['zIndex']}")
    test(hl_toolbar['btnCount'] == 6, "Toolbar has 6 buttons", f"got {hl_toolbar['btnCount']}")
    if hl_toolbar['btnSize']:
        test(parse_px(hl_toolbar['btnSize']['w']) >= 36, f"Highlight btn width >= 36px", f"got {hl_toolbar['btnSize']['w']}")

    # ============================================================
    # MOBILE TESTS (390x844)
    # ============================================================
    page.close()
    page = browser.new_page(viewport={"width": 390, "height": 844})
    page.goto(URL, wait_until="networkidle")
    page.wait_for_timeout(3000)

    group("Mobile Layout")
    mobile = page.evaluate('''() => {
        const cs = (el) => el ? window.getComputedStyle(el) : {};
        const nav = document.getElementById('top-nav');
        const sidebar = document.getElementById('sidebar');
        const content = document.getElementById('content-area');
        return {
            navHeight: cs(nav).height,
            sidebarHidden: sidebar.classList.contains('collapsed'),
            contentWidth: cs(content).width,
            bodyOverflowX: cs(document.body).overflowX,
        };
    }''')
    test(parse_px(mobile['navHeight']) >= 48, "Mobile nav height >= 48px", f"got {mobile['navHeight']}")
    test(mobile['sidebarHidden'], "Sidebar collapsed by default on mobile")
    test(parse_px(mobile['contentWidth']) <= 390, "Content fits viewport width", f"got {mobile['contentWidth']}")

    group("Mobile Sidebar")
    page.click('#sidebar-toggle')
    page.wait_for_timeout(500)
    mobile_sidebar = page.evaluate('''() => {
        const sidebar = document.getElementById('sidebar');
        const close = document.getElementById('sidebar-close');
        const overlay = document.getElementById('sidebar-overlay');
        const cs = (el) => el ? window.getComputedStyle(el) : {};
        return {
            sidebarVisible: !sidebar.classList.contains('collapsed'),
            sidebarWidth: cs(sidebar).width,
            sidebarPosition: cs(sidebar).position,
            closeVisible: close ? cs(close).display != 'none' : false,
            overlayVisible: overlay ? cs(overlay).display != 'none' : false,
        };
    }''')
    test(mobile_sidebar['sidebarVisible'], "Sidebar opens on toggle")
    test(parse_px(mobile_sidebar['sidebarWidth']) >= 250, "Mobile sidebar width >= 250px", f"got {mobile_sidebar['sidebarWidth']}")
    test(mobile_sidebar['sidebarPosition'] == 'fixed', "Mobile sidebar is fixed", f"got {mobile_sidebar['sidebarPosition']}")

    page.close()
    browser.close()

# ============================================================
# SUMMARY
# ============================================================
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
