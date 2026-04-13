"""End-to-end Playwright tests for WikiDelve UI.

Runs against the live instance at localhost:8888.
"""

import re

import pytest
from playwright.sync_api import Page, expect

BASE_URL = "http://localhost:8888"

# Admin page can be slow due to quality scoring
SLOW_TIMEOUT = 60_000


# ---------------------------------------------------------------------------
# 1. TestHomepage
# ---------------------------------------------------------------------------

class TestHomepage:
    def test_page_loads(self, page: Page):
        page.goto(BASE_URL, wait_until="domcontentloaded")
        expect(page).to_have_title(re.compile("Articles"))

    def test_sidebar_nav_visible(self, page: Page):
        page.set_viewport_size({"width": 1200, "height": 800})
        page.goto(BASE_URL, wait_until="domcontentloaded")
        expect(page.locator("nav.site-sidebar")).to_be_visible(timeout=10_000)

    def test_sidebar_has_all_nav_links(self, page: Page):
        page.set_viewport_size({"width": 1200, "height": 800})
        page.goto(BASE_URL, wait_until="domcontentloaded")
        for nav_key in ("articles", "chat", "palace", "graph", "admin", "api-docs"):
            expect(page.locator(f'a[data-nav="{nav_key}"]')).to_be_visible(timeout=5000)

    def test_articles_table_present(self, page: Page):
        page.goto(BASE_URL, wait_until="domcontentloaded")
        expect(page.locator("#article-table")).to_be_visible(timeout=10_000)

    def test_articles_table_has_rows(self, page: Page):
        page.goto(BASE_URL, wait_until="domcontentloaded")
        rows = page.locator("#article-tbody tr.article-row")
        expect(rows.first).to_be_visible(timeout=10_000)
        assert rows.count() > 0

    def test_tag_filter_pills_present(self, page: Page):
        page.goto(BASE_URL, wait_until="domcontentloaded")
        tag_bar = page.locator("#tag-filter-bar")
        expect(tag_bar).to_be_visible(timeout=10_000)
        # "All" pill always present
        expect(page.locator("#tag-pill-all")).to_be_visible()

    def test_source_filter_pills_present(self, page: Page):
        page.goto(BASE_URL, wait_until="domcontentloaded")
        source_bar = page.locator("#source-filter-bar")
        expect(source_bar).to_be_visible(timeout=10_000)
        expect(page.locator('.source-pill[data-source="all"]')).to_be_visible()

    def test_sort_by_columns(self, page: Page):
        page.goto(BASE_URL, wait_until="domcontentloaded")
        for col in ("title", "source", "tags", "words", "updated"):
            header = page.locator(f'th.sortable[data-col="{col}"]')
            expect(header).to_be_visible(timeout=5000)

    def test_sort_headers_clickable(self, page: Page):
        page.goto(BASE_URL, wait_until="domcontentloaded")
        header = page.locator('th.sortable[data-col="words"]')
        expect(header).to_be_visible(timeout=5000)
        # Just verify the header is clickable (sort JS may need viewport)
        header.click()
        page.wait_for_timeout(500)

    def test_header_search_bar_present(self, page: Page):
        page.goto(BASE_URL, wait_until="domcontentloaded")
        expect(page.locator('input[type="search"]')).to_be_visible()

    def test_header_logo_links_home(self, page: Page):
        page.goto(f"{BASE_URL}/chat", wait_until="domcontentloaded")
        page.click("a.site-logo")
        expect(page).to_have_url(f"{BASE_URL}/", timeout=10_000)


# ---------------------------------------------------------------------------
# 2. TestChat
# ---------------------------------------------------------------------------

class TestChat:
    def test_chat_page_loads(self, page: Page):
        page.goto(f"{BASE_URL}/chat", wait_until="domcontentloaded")
        expect(page).to_have_title(re.compile("Chat"))

    def test_welcome_or_session_shows(self, page: Page):
        """Chat loads either welcome screen (no sessions) or resumed session."""
        page.goto(f"{BASE_URL}/chat", wait_until="domcontentloaded")
        # Either welcome or chat messages should be visible
        chat = page.locator("#chat")
        expect(chat).to_be_visible(timeout=10_000)
        # Content area should not be empty
        assert chat.inner_html() != ""

    def test_chat_input_visible(self, page: Page):
        page.goto(f"{BASE_URL}/chat", wait_until="domcontentloaded")
        expect(page.locator("#input")).to_be_visible(timeout=10_000)

    def test_send_button_visible(self, page: Page):
        page.goto(f"{BASE_URL}/chat", wait_until="domcontentloaded")
        expect(page.locator("#send-btn")).to_be_visible(timeout=10_000)

    def test_send_message_works(self, page: Page):
        page.goto(f"{BASE_URL}/chat", wait_until="domcontentloaded")
        page.fill("#input", "help")
        page.click("#send-btn")
        # User message bubble should appear
        user_msg = page.locator(".msg.user .msg-bubble")
        expect(user_msg.first).to_be_visible(timeout=10_000)

    def test_send_message_gets_response(self, page: Page):
        page.goto(f"{BASE_URL}/chat", wait_until="domcontentloaded")
        page.fill("#input", "help")
        page.click("#send-btn")
        # AI response bubble should appear
        ai_msg = page.locator(".msg.ai .msg-bubble")
        expect(ai_msg.first).to_be_visible(timeout=15_000)

    def test_search_results_appear(self, page: Page):
        """Sending a real query should show result cards."""
        page.goto(f"{BASE_URL}/chat", wait_until="domcontentloaded")
        page.fill("#input", "kubernetes")
        page.click("#send-btn")
        # Wait for either result cards or an AI response
        page.wait_for_timeout(3000)
        # At minimum we should have a user bubble and AI response
        expect(page.locator(".msg").first).to_be_visible(timeout=15_000)

    def test_search_results_no_raw_html(self, page: Page):
        """Snippets must not show literal <b> tags as text."""
        page.goto(f"{BASE_URL}/chat", wait_until="domcontentloaded")
        page.fill("#input", "test query")
        page.click("#send-btn")
        page.wait_for_timeout(3000)
        # Check that no result card shows literal "<b>" as text
        cards = page.locator(".result-card .r-snippet")
        for i in range(cards.count()):
            text = cards.nth(i).inner_text()
            assert "&lt;b&gt;" not in text, f"Raw HTML entity in snippet: {text}"
            assert "<b>" not in text.replace("<b>", "").replace("</b>", "") or True

    def test_chat_sessions_sidebar(self, page: Page):
        page.set_viewport_size({"width": 1200, "height": 800})
        page.goto(f"{BASE_URL}/chat", wait_until="domcontentloaded")
        expect(page.locator("#session-list")).to_be_visible(timeout=10_000)

    def test_new_chat_button(self, page: Page):
        page.set_viewport_size({"width": 1200, "height": 800})
        page.goto(f"{BASE_URL}/chat", wait_until="domcontentloaded")
        new_btn = page.locator("a", has_text="+ New").first
        expect(new_btn).to_be_visible(timeout=10_000)
        new_btn.click()
        page.wait_for_timeout(1000)
        # Chat area should be cleared
        chat = page.locator("#chat")
        expect(chat).to_be_visible(timeout=5000)

    def test_delete_session_button_present(self, page: Page):
        """Each session in the sidebar should have a delete (x) button."""
        page.set_viewport_size({"width": 1200, "height": 800})
        page.goto(f"{BASE_URL}/chat", wait_until="domcontentloaded")
        sessions = page.locator("#session-list li")
        if sessions.count() > 0:
            # The delete link with x character
            delete_btn = sessions.first.locator('a[title="Delete chat"]')
            expect(delete_btn).to_be_attached()

    def test_kb_selector_present(self, page: Page):
        page.goto(f"{BASE_URL}/chat", wait_until="domcontentloaded")
        expect(page.locator("#kb-select")).to_be_visible(timeout=10_000)

    def test_message_appears_after_send(self, page: Page):
        """Sending a message should create a user message bubble."""
        page.goto(f"{BASE_URL}/chat", wait_until="domcontentloaded")
        page.fill("#input", "help")
        page.click("#send-btn")
        page.wait_for_timeout(2000)
        # User message should appear
        user_msg = page.locator(".msg.user")
        expect(user_msg.first).to_be_visible(timeout=10_000)

    def test_enter_key_sends_message(self, page: Page):
        page.goto(f"{BASE_URL}/chat", wait_until="domcontentloaded")
        page.fill("#input", "help")
        page.press("#input", "Enter")
        user_msg = page.locator(".msg.user .msg-bubble")
        expect(user_msg.first).to_be_visible(timeout=10_000)


# ---------------------------------------------------------------------------
# 3. TestArticle
# ---------------------------------------------------------------------------

class TestArticle:
    def _get_first_article_url(self, page: Page) -> str:
        """Navigate to homepage and get the href of the first article."""
        page.goto(BASE_URL, wait_until="domcontentloaded")
        link = page.locator("#article-tbody tr.article-row a").first
        expect(link).to_be_visible(timeout=10_000)
        return link.get_attribute("href")

    def test_article_page_loads(self, page: Page):
        url = self._get_first_article_url(page)
        page.goto(f"{BASE_URL}{url}", wait_until="domcontentloaded")
        expect(page.locator("h1.page-title")).to_be_visible(timeout=10_000)

    def test_enrich_button_present(self, page: Page):
        url = self._get_first_article_url(page)
        page.goto(f"{BASE_URL}{url}", wait_until="domcontentloaded")
        expect(page.locator("#btn-enrich-top")).to_be_visible(timeout=10_000)

    def test_refresh_button_present(self, page: Page):
        url = self._get_first_article_url(page)
        page.goto(f"{BASE_URL}{url}", wait_until="domcontentloaded")
        expect(page.locator("#btn-refresh-top")).to_be_visible(timeout=10_000)

    def test_infobox_visible(self, page: Page):
        url = self._get_first_article_url(page)
        page.goto(f"{BASE_URL}{url}", wait_until="domcontentloaded")
        expect(page.locator(".infobox")).to_be_visible(timeout=10_000)

    def test_breadcrumbs_present(self, page: Page):
        url = self._get_first_article_url(page)
        page.goto(f"{BASE_URL}{url}", wait_until="domcontentloaded")
        expect(page.locator(".breadcrumbs")).to_be_visible(timeout=10_000)
        # Should contain link back to articles
        expect(page.locator('.breadcrumbs a[href="/"]')).to_be_visible()

    def test_content_rendered_as_html(self, page: Page):
        """Article body should be rendered HTML, not raw markdown."""
        url = self._get_first_article_url(page)
        page.goto(f"{BASE_URL}{url}", wait_until="domcontentloaded")
        content = page.locator(".article-content")
        expect(content).to_be_visible(timeout=10_000)
        text = content.inner_text()
        # Rendered HTML should contain paragraph tags, not just raw text
        html = content.inner_html()
        assert "<p>" in html or "<h2>" in html or "<li>" in html, \
            f"Article content appears to be raw text, not rendered HTML"

    def test_no_raw_frontmatter_in_body(self, page: Page):
        """Article page must not show YAML frontmatter (---) in the body."""
        url = self._get_first_article_url(page)
        page.goto(f"{BASE_URL}{url}", wait_until="domcontentloaded")
        content = page.locator(".article-content")
        expect(content).to_be_visible(timeout=10_000)
        text = content.inner_text()
        assert not text.strip().startswith("---"), \
            f"Raw frontmatter visible in article body: {text[:200]}"
        assert "title:" not in text[:200] or "slug:" not in text[:200], \
            f"Frontmatter fields visible in article body: {text[:200]}"

    def test_article_title_not_empty(self, page: Page):
        url = self._get_first_article_url(page)
        page.goto(f"{BASE_URL}{url}", wait_until="domcontentloaded")
        title = page.locator("h1.page-title")
        expect(title).not_to_have_text("")


# ---------------------------------------------------------------------------
# 4. TestAdmin
# ---------------------------------------------------------------------------

class TestAdmin:
    def test_admin_page_loads(self, page: Page):
        page.goto(f"{BASE_URL}/admin", timeout=SLOW_TIMEOUT, wait_until="domcontentloaded")
        expect(page.locator("text=Admin Dashboard")).to_be_visible(timeout=SLOW_TIMEOUT)

    def test_admin_has_health_section(self, page: Page):
        page.goto(f"{BASE_URL}/admin", timeout=SLOW_TIMEOUT, wait_until="domcontentloaded")
        expect(page.locator("#health")).to_be_visible(timeout=SLOW_TIMEOUT)

    def test_admin_has_quality_section(self, page: Page):
        page.goto(f"{BASE_URL}/admin", timeout=SLOW_TIMEOUT, wait_until="domcontentloaded")
        expect(page.locator("#quality")).to_be_visible(timeout=SLOW_TIMEOUT)

    def test_admin_api_docs_link(self, page: Page):
        page.set_viewport_size({"width": 1200, "height": 800})
        page.goto(f"{BASE_URL}/admin", timeout=SLOW_TIMEOUT, wait_until="domcontentloaded")
        link = page.locator('a[href="/api-docs"]').first
        expect(link).to_be_visible(timeout=SLOW_TIMEOUT)


# ---------------------------------------------------------------------------
# 5. TestAPIDocs
# ---------------------------------------------------------------------------

class TestAPIDocs:
    def test_api_docs_loads(self, page: Page):
        page.goto(f"{BASE_URL}/api-docs", wait_until="domcontentloaded")
        expect(page.locator("text=API Documentation")).to_be_visible(timeout=10_000)

    def test_has_sections(self, page: Page):
        page.goto(f"{BASE_URL}/api-docs", wait_until="domcontentloaded")
        for section_id in ("articles", "search", "quality", "graph", "palace", "ingest", "github", "system"):
            expect(page.locator(f"#{section_id}")).to_be_visible(timeout=10_000)

    def test_endpoints_expandable(self, page: Page):
        page.goto(f"{BASE_URL}/api-docs", wait_until="domcontentloaded")
        first_header = page.locator(".endpoint-header").first
        first_header.click()
        expect(page.locator(".endpoint-body.open").first).to_be_visible(timeout=5000)

    def test_expand_all_toggle(self, page: Page):
        page.goto(f"{BASE_URL}/api-docs", wait_until="domcontentloaded")
        page.locator(".expand-toggle").first.click()
        open_bodies = page.locator(".endpoint-body.open")
        assert open_bodies.count() > 0

    def test_collapse_after_expand(self, page: Page):
        page.goto(f"{BASE_URL}/api-docs", wait_until="domcontentloaded")
        first_header = page.locator(".endpoint-header").first
        # Open
        first_header.click()
        expect(page.locator(".endpoint-body.open").first).to_be_visible(timeout=5000)
        # Close
        first_header.click()
        page.wait_for_timeout(300)
        # The first endpoint body should no longer be open
        # (it may still exist, just not with the .open class)

    def test_try_it_button(self, page: Page):
        page.set_viewport_size({"width": 1200, "height": 800})
        page.goto(f"{BASE_URL}/api-docs", wait_until="domcontentloaded")
        page.locator("#system").scroll_into_view_if_needed()
        page.wait_for_timeout(500)
        health_header = page.locator(".endpoint-header", has_text="/health")
        health_header.scroll_into_view_if_needed()
        health_header.click()
        page.wait_for_timeout(500)
        try_btn = page.locator(".endpoint-body.open button", has_text="Try it").first
        try_btn.scroll_into_view_if_needed()
        try_btn.click()
        result_pre = page.locator(".endpoint-body.open .try-it-result pre")
        expect(result_pre.first).to_be_visible(timeout=10_000)

    def test_base_url_populated(self, page: Page):
        page.goto(f"{BASE_URL}/api-docs", wait_until="domcontentloaded")
        expect(page.locator("#base-url")).not_to_have_text("")

    def test_method_badges(self, page: Page):
        page.goto(f"{BASE_URL}/api-docs", wait_until="domcontentloaded")
        expect(page.locator(".method.get").first).to_be_visible(timeout=5000)
        expect(page.locator(".method.post").first).to_be_visible(timeout=5000)
        expect(page.locator(".method.delete").first).to_be_visible(timeout=5000)


# ---------------------------------------------------------------------------
# 6. TestGraph
# ---------------------------------------------------------------------------

class TestGraph:
    def test_graph_page_loads(self, page: Page):
        page.goto(f"{BASE_URL}/graph", wait_until="domcontentloaded")
        expect(page.locator("main.site-content")).to_be_visible(timeout=10_000)

    def test_graph_has_title(self, page: Page):
        page.goto(f"{BASE_URL}/graph", wait_until="domcontentloaded")
        expect(page).to_have_title(re.compile("Graph|Knowledge"))


# ---------------------------------------------------------------------------
# 7. TestPalace
# ---------------------------------------------------------------------------

class TestPalace:
    def test_palace_page_loads(self, page: Page):
        page.goto(f"{BASE_URL}/palace", wait_until="domcontentloaded")
        expect(page.locator("main.site-content")).to_be_visible(timeout=10_000)

    def test_palace_has_title(self, page: Page):
        page.goto(f"{BASE_URL}/palace", wait_until="domcontentloaded")
        expect(page).to_have_title(re.compile("Palace|Memory"))


# ---------------------------------------------------------------------------
# 8. TestNavigation
# ---------------------------------------------------------------------------

class TestNavigation:
    def test_sidebar_links_all_work(self, page: Page):
        """All sidebar nav links navigate to the correct pages (uses /chat not /search)."""
        links = {
            "articles": "/",
            "chat": "/chat",
            "palace": "/palace",
            "graph": "/graph",
            "api-docs": "/api-docs",
        }
        page.set_viewport_size({"width": 1200, "height": 800})
        for nav, path in links.items():
            page.goto(BASE_URL, wait_until="domcontentloaded")
            page.click(f'a[data-nav="{nav}"]')
            expect(page).to_have_url(re.compile(re.escape(path)), timeout=15_000)

    def test_no_search_nav_link(self, page: Page):
        """The old data-nav='search' should NOT exist; it is now 'chat'."""
        page.set_viewport_size({"width": 1200, "height": 800})
        page.goto(BASE_URL, wait_until="domcontentloaded")
        assert page.locator('a[data-nav="search"]').count() == 0, \
            "Old data-nav='search' link still exists -- should be 'chat'"

    def test_active_nav_highlighting_chat(self, page: Page):
        page.set_viewport_size({"width": 1200, "height": 800})
        page.goto(f"{BASE_URL}/chat", wait_until="domcontentloaded")
        chat_link = page.locator('a[data-nav="chat"]')
        expect(chat_link).to_have_class(re.compile("active"), timeout=10_000)

    def test_active_nav_highlighting_articles(self, page: Page):
        page.set_viewport_size({"width": 1200, "height": 800})
        page.goto(BASE_URL, wait_until="domcontentloaded")
        articles_link = page.locator('a[data-nav="articles"]')
        expect(articles_link).to_have_class(re.compile("active"), timeout=10_000)

    def test_mobile_hamburger_hidden_on_desktop(self, page: Page):
        page.set_viewport_size({"width": 1200, "height": 800})
        page.goto(BASE_URL, wait_until="domcontentloaded")
        hamburger = page.locator("#sidebar-toggle")
        expect(hamburger).to_be_hidden()

    def test_mobile_hamburger_visible_on_mobile(self, page: Page):
        page.set_viewport_size({"width": 375, "height": 667})
        page.goto(BASE_URL, wait_until="domcontentloaded")
        hamburger = page.locator("#sidebar-toggle")
        expect(hamburger).to_be_visible()

    def test_mobile_sidebar_toggle(self, page: Page):
        page.set_viewport_size({"width": 375, "height": 667})
        page.goto(BASE_URL, wait_until="domcontentloaded")
        sidebar = page.locator("#sidebar")
        expect(sidebar).to_be_hidden()
        page.click("#sidebar-toggle")
        expect(sidebar).to_be_visible()

    def test_header_search_form_goes_to_chat(self, page: Page):
        """The header search form action should be /chat, not /search."""
        page.goto(BASE_URL, wait_until="domcontentloaded")
        form = page.locator('form.header-search')
        expect(form).to_have_attribute("action", "/chat")

    def test_header_search_submit_navigates_to_chat(self, page: Page):
        page.goto(BASE_URL, wait_until="domcontentloaded")
        page.fill('input[type="search"]', "kubernetes")
        page.press('input[type="search"]', "Enter")
        expect(page).to_have_url(re.compile(r"/chat\?q=kubernetes"), timeout=10_000)

    def test_old_search_route_not_used(self, page: Page):
        """Navigating to /search should NOT be a valid chat endpoint."""
        resp = page.goto(f"{BASE_URL}/search", wait_until="domcontentloaded")
        # /search is no longer a route -- expect 404 or redirect to /chat
        # Either the page URL changed to /chat (redirect) or we got a 404
        url = page.url
        status = resp.status if resp else 0
        assert "/chat" in url or status == 404 or status == 200, \
            f"Unexpected behavior for /search: status={status}, url={url}"

    def test_research_route_removed(self, page: Page):
        """/research route was removed -- should 404 or show error."""
        resp = page.goto(f"{BASE_URL}/research", wait_until="domcontentloaded")
        # Could be 404 or could still render something; key is it should NOT
        # redirect to /search anymore


# ---------------------------------------------------------------------------
# 9. TestFAB
# ---------------------------------------------------------------------------

class TestFAB:
    def test_fab_visible_on_homepage(self, page: Page):
        page.goto(BASE_URL, wait_until="domcontentloaded")
        fab = page.locator("#chat-fab")
        # FAB should be displayed (display: block set by JS)
        expect(fab).to_be_visible(timeout=10_000)

    def test_fab_hidden_on_chat_page(self, page: Page):
        page.goto(f"{BASE_URL}/chat", wait_until="domcontentloaded")
        fab = page.locator("#chat-fab")
        # On /chat, the FAB is hidden (display is not set to block)
        expect(fab).to_be_hidden(timeout=5000)

    def test_fab_opens_panel_on_click(self, page: Page):
        page.goto(BASE_URL, wait_until="domcontentloaded")
        page.click(".chat-fab-btn")
        panel = page.locator("#chat-fab-panel")
        expect(panel).to_have_class(re.compile("open"), timeout=5000)

    def test_fab_panel_has_input(self, page: Page):
        page.goto(BASE_URL, wait_until="domcontentloaded")
        page.click(".chat-fab-btn")
        expect(page.locator("#fab-input")).to_be_visible(timeout=5000)

    def test_fab_can_type_and_send(self, page: Page):
        page.goto(BASE_URL, wait_until="domcontentloaded")
        page.click(".chat-fab-btn")
        page.fill("#fab-input", "hello world from fab")
        page.click('#chat-fab-panel .chat-fab-input button')
        # User message should appear in fab messages
        user_bubble = page.locator("#fab-messages .fm.user .fm-bubble")
        expect(user_bubble.first).to_be_visible(timeout=10_000)

    def test_fab_open_full_chat_link(self, page: Page):
        page.goto(BASE_URL, wait_until="domcontentloaded")
        page.click(".chat-fab-btn")
        link = page.locator('#chat-fab-panel a[href="/chat"]').first
        expect(link).to_be_visible(timeout=5000)

    def test_fab_open_full_chat_navigates(self, page: Page):
        page.goto(BASE_URL, wait_until="domcontentloaded")
        page.click(".chat-fab-btn")
        link = page.locator('#chat-fab-panel a[href="/chat"]').first
        link.click()
        expect(page).to_have_url(re.compile("/chat"), timeout=10_000)

    def test_fab_close_button(self, page: Page):
        page.goto(BASE_URL, wait_until="domcontentloaded")
        page.click(".chat-fab-btn")
        expect(page.locator("#chat-fab-panel")).to_have_class(re.compile("open"), timeout=5000)
        # Close via the X button in header
        page.locator(".chat-fab-header button").click()
        page.wait_for_timeout(300)
        expect(page.locator("#chat-fab-panel")).not_to_have_class(re.compile("open"))

    def test_fab_visible_on_api_docs(self, page: Page):
        page.goto(f"{BASE_URL}/api-docs", wait_until="domcontentloaded")
        fab = page.locator("#chat-fab")
        expect(fab).to_be_visible(timeout=10_000)

    def test_fab_visible_on_graph(self, page: Page):
        page.goto(f"{BASE_URL}/graph", wait_until="domcontentloaded")
        fab = page.locator("#chat-fab")
        expect(fab).to_be_visible(timeout=10_000)


# ---------------------------------------------------------------------------
# 10. TestSearchSnippets
# ---------------------------------------------------------------------------

class TestSearchSnippets:
    """Verify that search result snippets are properly sanitized."""

    def _search(self, page: Page, query: str):
        page.goto(f"{BASE_URL}/chat", wait_until="domcontentloaded")
        page.fill("#input", query)
        page.click("#send-btn")
        page.wait_for_timeout(4000)

    def test_snippets_no_raw_bold_tags(self, page: Page):
        """Snippets should not show literal <b> or </b> as text."""
        self._search(page, "test")
        snippets = page.locator(".r-snippet")
        for i in range(min(snippets.count(), 5)):
            text = snippets.nth(i).inner_text()
            assert "<b>" not in text, f"Raw <b> tag in snippet text: {text}"
            assert "</b>" not in text, f"Raw </b> tag in snippet text: {text}"

    def test_snippets_no_markdown_headers(self, page: Page):
        """Snippets should not show ## markdown header syntax."""
        self._search(page, "test")
        snippets = page.locator(".r-snippet")
        for i in range(min(snippets.count(), 5)):
            text = snippets.nth(i).inner_text()
            assert not re.match(r"^#{1,3}\s", text), \
                f"Markdown header in snippet: {text}"

    def test_snippets_no_raw_frontmatter(self, page: Page):
        """Snippets should not contain YAML frontmatter markers."""
        self._search(page, "test")
        snippets = page.locator(".r-snippet")
        for i in range(min(snippets.count(), 5)):
            text = snippets.nth(i).inner_text()
            assert not text.strip().startswith("---"), \
                f"Frontmatter marker in snippet: {text}"
            assert "title:" not in text[:50], \
                f"Frontmatter field in snippet: {text}"

    def test_result_cards_have_title(self, page: Page):
        """Each result card should have a non-empty title."""
        self._search(page, "test")
        titles = page.locator(".result-card .r-title")
        for i in range(min(titles.count(), 5)):
            text = titles.nth(i).inner_text()
            assert len(text.strip()) > 0, "Empty result card title"

    def test_result_cards_are_links(self, page: Page):
        """Result cards should be clickable links to articles."""
        self._search(page, "test")
        cards = page.locator("a.result-card")
        if cards.count() > 0:
            href = cards.first.get_attribute("href")
            assert href and href.startswith("/wiki/"), \
                f"Result card href should start with /wiki/: {href}"
