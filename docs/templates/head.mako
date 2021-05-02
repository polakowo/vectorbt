<%!
    from pdoc.html_helpers import minify_css
%>
<%def name="homelink()" filter="minify_css">
    .homelink {
        display: block;
        font-size: 2em;
        font-weight: bold;
        color: white;
    }
    .homelink:hover {
        color: #e82;
    }
    .homelink img {
        max-width: 100px;
        max-height: 100px;
        margin: auto;
        margin-bottom: .3em;
    }
</%def>

<style>${homelink()}</style>
<link rel="apple-touch-icon" sizes="180x180" href="https://raw.githubusercontent.com/polakowo/vectorbt/master/static/favicon/apple-touch-icon.png">
<link rel="icon" type="image/png" sizes="32x32" href="https://raw.githubusercontent.com/polakowo/vectorbt/master/static/favicon/favicon-32x32.png">
<link rel="icon" type="image/png" sizes="16x16" href="https://raw.githubusercontent.com/polakowo/vectorbt/master/static/favicon/favicon-16x16.png">
<link rel="manifest" href="https://raw.githubusercontent.com/polakowo/vectorbt/master/static/favicon/site.webmanifest">
<link rel="icon" href="https://raw.githubusercontent.com/polakowo/vectorbt/master/static/favicon/favicon.ico">
<meta name="msapplication-TileColor" content="#282c34">
<meta name="theme-color" content="#282c34">
