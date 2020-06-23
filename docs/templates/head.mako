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
<link rel="icon" href="https://github.com/polakowo/vectorbt/blob/master/logo.png?raw=true">