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
        max-width: 128px;
        max-height: 128px;
        margin: auto;
        margin-bottom: .3em;
    }
</%def>

<style>${homelink()}</style>
<link rel="apple-touch-icon" sizes="180x180" href="https://raw.githubusercontent.com/polakowo/vectorbt/master/static/favicon/apple-touch-icon.png">
<link rel="icon" type="image/png" sizes="32x32" href="https://raw.githubusercontent.com/polakowo/vectorbt/master/static/favicon/favicon-32x32.png">
<link rel="icon" type="image/png" sizes="16x16" href="https://raw.githubusercontent.com/polakowo/vectorbt/master/static/favicon/favicon-16x16.png">
<link rel="manifest" href="https://raw.githubusercontent.com/polakowo/vectorbt/master/static/favicon/site.webmanifest">
<link rel="mask-icon" href="https://raw.githubusercontent.com/polakowo/vectorbt/master/static/favicon/safari-pinned-tab.svg" color="#5bbad5">
<meta name="msapplication-TileColor" content="#da532c">
<meta name="theme-color" content="#ffffff">
